from pydantic import BaseModel
from typing import List, Tuple, Dict
from shapely import Polygon, MultiPolygon, get_coordinates, unary_union
from math import isnan

from geolib.models.dstability import DStabilityModel
from geolib.soils.soil import Soil
from geolib.geometry.one import Point
from geolib.models.dstability.internal import (
    AnalysisTypeEnum,
    CharacteristicPointEnum,
    ShearStrengthModelTypePhreaticLevelInternal,
)
from geolib.models.dstability.analysis import (
    DStabilityBishopBruteForceAnalysisMethod,
    DStabilitySearchGrid,
    DStabilitySpencerGeneticAnalysisMethod,
    DStabilityUpliftVanParticleSwarmAnalysisMethod,
    DStabilitySearchArea,
    DStabilitySlipPlaneConstraints,
    DStabilityGeneticSlipPlaneConstraints,
)


class SoilPolygon(BaseModel):
    points: List[Tuple[float, float]] = []
    soilcode: str

    @property
    def shapely_polygon(self) -> Polygon:
        return Polygon(self.points)


class DStabilityModelModifier(BaseModel):
    dm: DStabilityModel = DStabilityModel()
    soils: List[Soil] = []
    soil_polygons: List[SoilPolygon] = []
    phreatic_line: List[Tuple[float, float]] = []
    scenario_index: int = 0
    stage_index: int = 0
    calculation_index: int = 0
    # phreatic_line_offset: float = 0.25
    # x_top_water_side: float = None
    # polder_level: float

    @classmethod
    def from_file(
        cls,
        filename: str,
    ) -> "DStabilityModelModifier":
        """Create a DStabilityModelModifier from a stix file

        Args:
            filename (str): The file to open

        Returns:
            DStabilityModelModifier: The DStabilityModifier based on the given file
        """
        result = DStabilityModelModifier(
            # x_top_water_side=x_top_water_side, polder_level=polder_level
        )
        result.dm.parse(filename)
        result.initialize()
        return result

    def initialize(self):
        """Update the model from the given DStabilityModel"""
        geometry = self.dm._get_geometry(
            scenario_index=self.scenario_index, stage_index=self.stage_index
        )

        soils = self._get_soils()

        for psoil in self.dm.soils.Soils:
            soil = self.dm.soils.get_global_soil(psoil.Code)

            if (
                psoil.ShearStrengthModelTypeBelowPhreaticLevel
                == ShearStrengthModelTypePhreaticLevelInternal.MOHR_COULOMB_CLASSIC
            ):
                if isnan(psoil.MohrCoulombClassicShearStrengthModel.Cohesion):
                    continue

                soil.soil_weight_parameters.unsaturated_weight = (
                    psoil.VolumetricWeightAbovePhreaticLevel
                )
                soil.soil_weight_parameters.saturated_weight = (
                    psoil.VolumetricWeightBelowPhreaticLevel
                )
                soil.mohr_coulomb_parameters.cohesion = (
                    psoil.MohrCoulombClassicShearStrengthModel.Cohesion
                )
                soil.mohr_coulomb_parameters.friction_angle = (
                    psoil.MohrCoulombClassicShearStrengthModel.FrictionAngle
                )
                # assume the friction angle equals to the friction angle
                soil.mohr_coulomb_parameters.dilatancy_angle = (
                    psoil.MohrCoulombClassicShearStrengthModel.FrictionAngle
                )
            elif (
                psoil.ShearStrengthModelTypeBelowPhreaticLevel
                == ShearStrengthModelTypePhreaticLevelInternal.MOHR_COULOMB_ADVANCED
            ):
                if isnan(psoil.MohrCoulombAdvancedShearStrengthModel.Cohesion):
                    continue

                soil.soil_weight_parameters.unsaturated_weight = (
                    psoil.VolumetricWeightAbovePhreaticLevel
                )
                soil.soil_weight_parameters.saturated_weight = (
                    psoil.VolumetricWeightBelowPhreaticLevel
                )
                soil.mohr_coulomb_parameters.cohesion = (
                    psoil.MohrCoulombAdvancedShearStrengthModel.Cohesion
                )
                soil.mohr_coulomb_parameters.friction_angle = (
                    psoil.MohrCoulombAdvancedShearStrengthModel.FrictionAngle
                )
                # assume the friction angle equals to the friction angle
                soil.mohr_coulomb_parameters.dilatancy_angle = (
                    psoil.MohrCoulombAdvancedShearStrengthModel.Dilatancy
                )
            self.soils.append(soil)

        # get the phreatic line
        if self.dm.phreatic_line is not None:
            self.phreatic_line = [(p.X, p.Z) for p in self.dm.phreatic_line().Points]

        # grondlaag connecties met grondsoorten
        layer_soil_dict = {
            l.LayerId: l.SoilId
            for l in self.dm._get_soil_layers(
                scenario_index=self.scenario_index, stage_index=self.stage_index
            ).SoilLayers
        }

        self.soil_polygons = [
            SoilPolygon(
                points=[(p.X, p.Z) for p in layer.Points],
                soilcode=soils[layer_soil_dict[layer.Id]]["code"],
            )
            for layer in geometry.Layers
        ]

    def set_scenario_stage_calculation_index(
        self, scenario_index: int = 0, stage_index: int = 0, calculation_index: int = 0
    ):
        """Set the indices of the scenario, stage and/or calculation, this will reset any modifications and
        there is no check if the values are valid

        Args:
            scenario_index (int, optional): The scenarion index to use. Defaults to 0.
            stage_index (int, optional): The stage index to use. Defaults to 0.
            calculation_index (int, optional): The calculation index to use. Defaults to 0.
        """
        self.scenario_index = scenario_index
        self.stage_index = stage_index
        self.calculation_index = calculation_index
        self.initialize()

    def cut(self, line: List[Tuple[float, float]], adjust_phreatic_line: bool = True):
        """Cut from a given line, the line should start above the surface. A point will be
        added to the line with a z value of 1m above the surface or the z value of the first points
        to finish the polygon that will be subtracted from the geometry

        Args:
            line (List[Tuple[float, float]]): Points of the line to cut
            adjust_phreatic_line (bool, optional): Adjust the phreatic line to the new geometry. Defaults to False.

        Raises:
            ValueError: Will raise an error if an invalid polygon calculation is found
        """
        zmax = max(line[0][1], max([p[1] for p in self.dm.surface]) + 1.0)

        cut_line = [p for p in line]
        cut_line.append((line[-1][0], zmax))
        cut_line.append((line[0][0], zmax))

        pg_extract = Polygon(cut_line)
        new_soil_polygons = []
        for spg in self.soil_polygons:
            pg = spg.shapely_polygon

            pgs = pg.difference(pg_extract)

            if type(pgs) == MultiPolygon:
                geoms = pgs.geoms
            elif type(pgs) == Polygon:
                geoms = [pgs]
            else:
                raise ValueError(f"Unhandled polygon difference type '{type(pgs)}'")

            for geom in geoms:
                points = get_coordinates(geom).tolist()
                new_soil_polygons.append(
                    SoilPolygon(points=points, soilcode=spg.soilcode)
                )

        self.soil_polygons = new_soil_polygons

    def fill(self, line: List[Tuple[float, float]], soil_code: str):
        """Fill the geometry from the given line with the given material. The line should be defined from left
        to right. A point will be added to the start and end of the line if that point is above the surface

        Args:
            soil_code (str): The soilcode to use for the fill material (should already be present in the availabale soils)
            line (List[Tuple[float, float]]): The line to use for the fill


        Raises:
            ValueError: Raises an error if the soilcode is not found or if an invalid polygon calculation is found
        """
        # TEST
        fill_line = [p for p in line]

        # add the soil unless it is already available
        if not soil_code in [s.code for s in self.soils]:
            raise ValueError(f"Soilcode '{soil_code}' for fill not found. Add it first")

        merged_polygon = unary_union(
            [spg.shapely_polygon for spg in self.soil_polygons]
        )

        # now create a polygon for the fill layer
        # we use the lowest point on the surface as the bottom of the line
        xmin = min([p[0] for p in line])
        xmax = max([p[0] for p in line])

        surfacepoints = [p for p in self.dm.surface if p[0] >= xmin and p[0] <= xmax]
        zmin = min([p[1] for p in surfacepoints]) - 0.1  # add small offset

        if fill_line[-1][1] > zmin:
            fill_line.append((fill_line[-1][0], zmin))
        if line[0][-1] > zmin:
            fill_line.append((fill_line[0][0], zmin))

        pgfill = Polygon(fill_line)

        pgs = pgfill.difference(merged_polygon)

        if type(pgs) == MultiPolygon:
            geoms = pgs.geoms
        elif type(pgs) == Polygon:
            geoms = [pgs]
        else:
            raise ValueError(f"Unhandled polygon difference type '{type(pgs)}'")

        for geom in geoms:
            if (
                geom.area > 0.1
            ):  # it is possible that really small layers will be added because of rounding errors, by checking the area for a min size we avoid these layers
                points = list(geom.exterior.coords)[:-1]
                self.soil_polygons.append(
                    SoilPolygon(points=points, soilcode=soil_code)
                )

        # if adjust_phreatic_line:
        #     self.adjust_phreatic_level()

    def to_dstability_model_with_autogenerated_settings(
        self,
        point_ref: Tuple[float, float],
        point_crest_land: Tuple[float, float],
        point_toe: Tuple[float, float],
        ditch_points: List[Tuple[float, float]],
    ):
        dm = self.to_dstability_model()
        # generate calculation settings
        settings = self.dm._get_calculation_settings(
            self.scenario_index, self.calculation_index
        )

        if (
            settings is not None
            or settings.AnalysisType == AnalysisTypeEnum.BISHOP_BRUTE_FORCE
        ):
            dm.set_model(
                DStabilityBishopBruteForceAnalysisMethod(
                    search_grid=DStabilitySearchGrid(
                        bottom_left=Point(
                            x=(point_ref[0] + point_crest_land[0]) / 2.0,
                            z=dm.z_at((point_ref[0] + point_crest_land[0]) / 2.0) + 1.0,
                        ),
                        number_of_points_in_x=20,
                        number_of_points_in_z=10,
                        space=0.5,
                    ),
                    bottom_tangent_line_z=point_toe[1] - 5.0,
                    number_of_tangent_lines=10,
                    space_tangent_lines=0.5,
                    slip_plane_constraints=DStabilitySlipPlaneConstraints(
                        is_size_constraints_enabled=True,
                        # is_zone_a_constraints_enabled=settings.BishopBruteForce.SlipPlaneConstraints.IsZoneAConstraintsEnabled,
                        # is_zone_b_constraints_enabled=settings.BishopBruteForce.SlipPlaneConstraints.IsZoneBConstraintsEnabled,
                        minimum_slip_plane_depth=2.0,
                        minimum_slip_plane_length=3.0,
                        # width_zone_a=settings.BishopBruteForce.SlipPlaneConstraints.WidthZoneA,
                        # width_zone_b=settings.BishopBruteForce.SlipPlaneConstraints.WidthZoneB,
                        # x_left_zone_a=settings.BishopBruteForce.SlipPlaneConstraints.XLeftZoneA,
                        # x_left_zone_b=settings.BishopBruteForce.SlipPlaneConstraints.XLeftZoneB,
                    ),
                )
            )
        elif settings.AnalysisType == AnalysisTypeEnum.SPENCER_GENETIC:
            dm.set_model(
                DStabilityBishopBruteForceAnalysisMethod(
                    search_grid=DStabilitySearchGrid(
                        bottom_left=Point(
                            x=(point_ref[0] + point_crest_land[0]) / 2.0,
                            z=dm.z_at((point_ref[0] + point_crest_land[0]) / 2.0) + 1.0,
                        ),
                        number_of_points_in_x=20,
                        number_of_points_in_z=10,
                        space=0.5,
                    ),
                    bottom_tangent_line_z=point_crest_land[1],
                    number_of_tangent_lines=10,
                    space_tangent_lines=0.5,
                    slip_plane_constraints=DStabilitySlipPlaneConstraints(
                        is_size_constraints_enabled=True,
                        # is_zone_a_constraints_enabled=settings.BishopBruteForce.SlipPlaneConstraints.IsZoneAConstraintsEnabled,
                        # is_zone_b_constraints_enabled=settings.BishopBruteForce.SlipPlaneConstraints.IsZoneBConstraintsEnabled,
                        minimum_slip_plane_depth=2.0,
                        minimum_slip_plane_length=3.0,
                        # width_zone_a=settings.BishopBruteForce.SlipPlaneConstraints.WidthZoneA,
                        # width_zone_b=settings.BishopBruteForce.SlipPlaneConstraints.WidthZoneB,
                        # x_left_zone_a=settings.BishopBruteForce.SlipPlaneConstraints.XLeftZoneA,
                        # x_left_zone_b=settings.BishopBruteForce.SlipPlaneConstraints.XLeftZoneB,
                    ),
                )
            )
        elif settings.AnalysisType == AnalysisTypeEnum.UPLIFT_VAN_PARTICLE_SWARM:
            dm.set_model(
                DStabilityBishopBruteForceAnalysisMethod(
                    search_grid=DStabilitySearchGrid(
                        bottom_left=Point(
                            x=(point_ref[0] + point_crest_land[0]) / 2.0,
                            z=dm.z_at((point_ref[0] + point_crest_land[0]) / 2.0) + 1.0,
                        ),
                        number_of_points_in_x=20,
                        number_of_points_in_z=10,
                        space=0.5,
                    ),
                    bottom_tangent_line_z=point_crest_land[1],
                    number_of_tangent_lines=10,
                    space_tangent_lines=0.5,
                    slip_plane_constraints=DStabilitySlipPlaneConstraints(
                        is_size_constraints_enabled=True,
                        # is_zone_a_constraints_enabled=settings.BishopBruteForce.SlipPlaneConstraints.IsZoneAConstraintsEnabled,
                        # is_zone_b_constraints_enabled=settings.BishopBruteForce.SlipPlaneConstraints.IsZoneBConstraintsEnabled,
                        minimum_slip_plane_depth=2.0,
                        minimum_slip_plane_length=3.0,
                        # width_zone_a=settings.BishopBruteForce.SlipPlaneConstraints.WidthZoneA,
                        # width_zone_b=settings.BishopBruteForce.SlipPlaneConstraints.WidthZoneB,
                        # x_left_zone_a=settings.BishopBruteForce.SlipPlaneConstraints.XLeftZoneA,
                        # x_left_zone_b=settings.BishopBruteForce.SlipPlaneConstraints.XLeftZoneB,
                    ),
                )
            )
        return dm

    def to_dstability_model(self) -> DStabilityModel:
        """Generate a DStability model from the modifier

        Returns:
            DStabilityModel: A DStabilityModel
        """
        dm = DStabilityModel()

        # copy the soils
        # NOTE this will only work for the MC model
        # soils that have a different model are not copied
        for soil in self.soils:
            if not dm.soils.has_soil_code(soil.code):
                dm.add_soil(soil)

        # create the layers
        for spg in self.soil_polygons:
            points = [Point(x=p[0], z=p[1]) for p in spg.points]
            dm.add_layer(points, soil_code=spg.soilcode)

            # create the phreatic line
            if len(self.phreatic_line) > 0:
                dm.add_head_line(
                    points=[Point(x=p[0], z=p[1]) for p in self.phreatic_line],
                    is_phreatic_line=True,
                    label="PL 1",
                )

        # create the phreatic line
        dm.add_head_line(
            points=[Point(x=p[0], z=p[1]) for p in self.phreatic_line],
            label="PL 1",
            is_phreatic_line=True,
        )

        #################################################
        # THE NEXT CODE WILL COPY THE ORIGINAL SETTINGS #
        #################################################
        settings = self.dm._get_calculation_settings(
            self.scenario_index, self.calculation_index
        )

        if settings is not None:
            if settings.AnalysisType == AnalysisTypeEnum.BISHOP_BRUTE_FORCE:
                dm.set_model(
                    DStabilityBishopBruteForceAnalysisMethod(
                        search_grid=DStabilitySearchGrid(
                            bottom_left=Point(
                                x=settings.BishopBruteForce.SearchGrid.BottomLeft.X,
                                z=settings.BishopBruteForce.SearchGrid.BottomLeft.Z,
                            ),
                            number_of_points_in_x=settings.BishopBruteForce.SearchGrid.NumberOfPointsInX,
                            number_of_points_in_z=settings.BishopBruteForce.SearchGrid.NumberOfPointsInZ,
                            space=settings.BishopBruteForce.SearchGrid.Space,
                        ),
                        bottom_tangent_line_z=settings.BishopBruteForce.TangentLines.BottomTangentLineZ,
                        number_of_tangent_lines=settings.BishopBruteForce.TangentLines.NumberOfTangentLines,
                        space_tangent_lines=settings.BishopBruteForce.TangentLines.Space,
                        slip_plane_constraints=DStabilitySlipPlaneConstraints(
                            is_size_constraints_enabled=settings.BishopBruteForce.SlipPlaneConstraints.IsSizeConstraintsEnabled,
                            is_zone_a_constraints_enabled=settings.BishopBruteForce.SlipPlaneConstraints.IsZoneAConstraintsEnabled,
                            is_zone_b_constraints_enabled=settings.BishopBruteForce.SlipPlaneConstraints.IsZoneBConstraintsEnabled,
                            minimum_slip_plane_depth=settings.BishopBruteForce.SlipPlaneConstraints.MinimumSlipPlaneDepth,
                            minimum_slip_plane_length=settings.BishopBruteForce.SlipPlaneConstraints.MinimumSlipPlaneLength,
                            width_zone_a=settings.BishopBruteForce.SlipPlaneConstraints.WidthZoneA,
                            width_zone_b=settings.BishopBruteForce.SlipPlaneConstraints.WidthZoneB,
                            x_left_zone_a=settings.BishopBruteForce.SlipPlaneConstraints.XLeftZoneA,
                            x_left_zone_b=settings.BishopBruteForce.SlipPlaneConstraints.XLeftZoneB,
                        ),
                    )
                )
            elif settings.AnalysisType == AnalysisTypeEnum.SPENCER_GENETIC:
                dm.set_model(
                    DStabilitySpencerGeneticAnalysisMethod(
                        slip_plane_a=[
                            Point(x=p.X, z=p.Z)
                            for p in settings.SpencerGenetic.SlipPlaneA
                        ],
                        slip_plane_b=[
                            Point(x=p.X, z=p.Z)
                            for p in settings.SpencerGenetic.SlipPlaneB
                        ],
                        slip_plane_constraints=DStabilityGeneticSlipPlaneConstraints(
                            is_enabled=settings.SpencerGenetic.SlipPlaneConstraints.IsEnabled,
                            minimum_angle_between_slices=settings.SpencerGenetic.SlipPlaneConstraints.MinimumAngleBetweenSlices,
                            minimum_thrust_line_percentage_inside_slices=settings.SpencerGenetic.SlipPlaneConstraints.MinimumThrustLinePercentageInsideSlices,
                        ),
                    )
                )
            elif settings.AnalysisType == AnalysisTypeEnum.UPLIFT_VAN_PARTICLE_SWARM:
                dm.set_model(
                    DStabilityUpliftVanParticleSwarmAnalysisMethod(
                        search_area_a=DStabilitySearchArea(
                            height=settings.UpliftVanParticleSwarm.SearchAreaA.Height,
                            top_left=Point(
                                x=settings.UpliftVanParticleSwarm.SearchAreaA.TopLeft.X,
                                z=settings.UpliftVanParticleSwarm.SearchAreaA.TopLeft.Z,
                            ),
                            width=settings.UpliftVanParticleSwarm.SearchAreaA.Width,
                        ),
                        search_area_b=DStabilitySearchArea(
                            height=settings.UpliftVanParticleSwarm.SearchAreaB.Height,
                            top_left=Point(
                                x=settings.UpliftVanParticleSwarm.SearchAreaB.TopLeft.X,
                                z=settings.UpliftVanParticleSwarm.SearchAreaB.TopLeft.Z,
                            ),
                            width=settings.UpliftVanParticleSwarm.SearchAreaB.Width,
                        ),
                        tangent_area_height=settings.UpliftVanParticleSwarm.TangentArea.Height,
                        tangent_area_top_z=settings.UpliftVanParticleSwarm.TangentArea.TopZ,
                        slip_plane_constraints=DStabilitySlipPlaneConstraints(
                            is_size_constraints_enabled=settings.UpliftVanParticleSwarm.SlipPlaneConstraints.IsSizeConstraintsEnabled,
                            is_zone_a_constraints_enabled=settings.UpliftVanParticleSwarm.SlipPlaneConstraints.IsZoneAConstraintsEnabled,
                            is_zone_b_constraints_enabled=settings.UpliftVanParticleSwarm.SlipPlaneConstraints.IsZoneBConstraintsEnabled,
                            minimum_slip_plane_depth=settings.UpliftVanParticleSwarm.SlipPlaneConstraints.MinimumSlipPlaneDepth,
                            minimum_slip_plane_length=settings.UpliftVanParticleSwarm.SlipPlaneConstraints.MinimumSlipPlaneLength,
                            width_zone_a=settings.UpliftVanParticleSwarm.SlipPlaneConstraints.WidthZoneA,
                            width_zone_b=settings.UpliftVanParticleSwarm.SlipPlaneConstraints.WidthZoneB,
                            x_left_zone_a=settings.UpliftVanParticleSwarm.SlipPlaneConstraints.XLeftZoneA,
                            x_left_zone_b=settings.UpliftVanParticleSwarm.SlipPlaneConstraints.XLeftZoneB,
                        ),
                    )
                )

        return dm

    def reset(self):
        """Reset all adjustments"""
        self.initialize()

    def set_phreatic_line(self, points: List[Tuple[float, float]]):
        self.phreatic_line = points

    # def adjust_phreatic_level(
    #     self,
    #     adjust_ditch: bool = False,
    #     ditch_points: List[Tuple[float, float]] = [],
    # ):
    #     """Adjust the phreatic line to the new surface, if ditch points are defined in the Waternet creator this will
    #     follow the ditch and allow the waterline to be above the surface between the ditch boundaries

    #     Args:
    #         x_top_water_side (float, optional): The x coordinate of the top of the levee on the water side, if not given this will be extracted from the Waternet creator settings. Before this coordinate the waterline can be above the surface level
    #         ditch_points (List[Tuple[float, float]], optional): Used defined ditch points. Defaults to []

    #     Raises:
    #         ValueError: If no x_top_water_side is defined and it cannot be found in the Waternet Creator settings an error will be raised
    #     """
    #     temp_dm = self.to_dstability_model()
    #     new_surface = [p for p in temp_dm.surface if p[0] > self.x_top_water_side]

    #     if len(self.dm.ditch_points) > 0 and len(ditch_points) == 0:
    #         ditch_points = self.dm.ditch_points

    #     if self.x_top_water_side is None:
    #         raise ValueError(
    #             "This model does not have a waternet creator setting for the embankment top water side which is needed for the phreatic line algorithm to work."
    #         )

    #     # copy all points before x_top_water_side
    #     points = [
    #         p
    #         for p in [
    #             (p.X, p.Z)
    #             for p in self.dm.phreatic_line(
    #                 self.scenario_index, self.stage_index
    #             ).Points
    #         ]
    #         if p[0] <= self.x_top_water_side
    #     ]

    #     ditch_added = False
    #     for p in new_surface:
    #         # use last points z coord
    #         pl_z = points[-1][1]
    #         surface_z = temp_dm.z_at(p[0])

    #         if surface_z is None:
    #             continue

    #         pl_z = min(pl_z, surface_z - self.phreatic_line_offset)

    #         # but keep the offset from the surface line unless we are at the ditch
    #         if len(ditch_points) > 0 and adjust_ditch:
    #             if ditch_points[0][0] <= p[0] and p[0] <= ditch_points[-1][0]:
    #                 if not ditch_added:
    #                     points.append((ditch_points[0][0], pl_z))
    #                     points.append((ditch_points[1][0], self.polder_level))
    #                     points.append((ditch_points[2][0], self.polder_level))
    #                     points.append((ditch_points[3][0], pl_z))
    #                     ditch_added = True
    #                 else:
    #                     continue
    #         else:
    #             points.append((p[0], pl_z))

    #     self.phreatic_line = points

    def _get_soils(self) -> List[Dict]:
        """Get the soils in the DStabilityModel as a list of dictionaries with the SoilId as the key

        Returns:
            List[Dict]: A list of soil dictionaries with the SoilId as key
        """
        return {
            s.Id: {
                "code": s.Code,
                "ys": s.VolumetricWeightBelowPhreaticLevel,
                "cohesion": s.MohrCoulombClassicShearStrengthModel.Cohesion,
            }
            for s in self.dm.datastructure.soils.Soils
        }


if __name__ == "__main__":
    dsm = DStabilityModelModifier.from_file("testdata/01.stix")
    dm_bishop = dsm.to_dstability_model()
    dm_bishop.serialize("output_bishop.stix")
    dsm.set_scenario_stage_calculation_index(scenario_index=1)
    dm_spencer = dsm.to_dstability_model()
    dm_spencer.serialize("output_spencer.stix")
    dsm.set_scenario_stage_calculation_index(scenario_index=2)
    dm_uplift = dsm.to_dstability_model()
    dm_uplift.serialize("output_uplift.stix")
    # back to bishop
    dsm.set_scenario_stage_calculation_index(scenario_index=0)
    dsm.fill(
        line=[(8.0, -1.0), (14.0, -1.5), (18.0, -3.5)],
        soil_code="Material_K[10.8-13.5]",
    )
    dm_bishop_with_fill = dsm.to_dstability_model()
    dm_bishop_with_fill.serialize("output_bishop_with_fill.stix")
    # reset for
    dsm.reset()
    dsm.cut(line=[(4.0, 1.0), (10.0, -5.5), (26.0, -7.5)], adjust_phreatic_line=True)
    dm_bishop_with_cut = dsm.to_dstability_model()
    dm_bishop_with_cut.serialize("output_bishop_with_cut.stix")
