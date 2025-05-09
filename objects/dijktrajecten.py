from .onderhoudsdieptes import OnderhoudsDieptes


class Dijktrajecten:
    def __init__(self):
        self.items = []

    @classmethod
    def from_csv(cls, csv_file, csv_file_onderhoudsdiepte):
        result = Dijktrajecten()
        dth_lines = open(csv_file, "r").readlines()
        onderhoudsdieptes = OnderhoudsDieptes.from_csv(csv_file_onderhoudsdiepte)

        for line in dth_lines[1:]:
            args = [a.strip() for a in line.split(",")]
            code = args[0]
            start = float(args[1])
            end = float(args[2])
            mhw = float(args[3])
            mhw_2024 = float(args[4])
            dth_2024 = float(args[5])
            onderhoudsdiepte = onderhoudsdieptes.get(code.lower(), (start + end) / 2.0)
            if result.get_by_code(code) is None:
                result.items.append(Dijktraject(code=code))
            result.get_by_code(code).add_part(
                start, end, mhw, mhw_2024, dth_2024, onderhoudsdiepte
            )

        return result

    def get_by_code(self, code):
        for i in self.items:
            if i.code == code:
                return i

        return None


class DijktrajectPart:
    def __init__(self, start, end, mhw, mhw_2024, dth_2024, onderhoudsdiepte):
        self.start = start
        self.end = end
        self.mhw = mhw
        self.mhw_2024 = mhw_2024
        self.dth_2024 = dth_2024
        self.onderhoudsdiepte = onderhoudsdiepte


class Dijktraject:
    def __init__(self, code):
        self.code = code
        self.parts = []

    def add_part(self, start, end, mhw, mhw_2024, dth_2024, onderhoudsdiepte):
        self.parts.append(
            DijktrajectPart(start, end, mhw, mhw_2024, dth_2024, onderhoudsdiepte)
        )

    def dth_2024_at(self, l: float):
        for part in self.parts:
            if part.start <= l and l <= part.end:
                return part.dth_2024

        return None

    def mhw_2024_at(self, l: float):
        for part in self.parts:
            if part.start <= l and l <= part.end:
                return part.mhw_2024

        return None

    def onderhoudsdiepte_at(self, l: float):
        for part in self.parts:
            if part.start <= l and l <= part.end:
                return part.onderhoudsdiepte

        return None
