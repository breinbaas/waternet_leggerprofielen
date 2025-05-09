from typing import Optional


class OnderhoudsDieptes:
    def __init__(self):
        self.items = []

    @classmethod
    def from_csv(cls, csv_file):
        result = OnderhoudsDieptes()
        lines = open(csv_file, "r").readlines()[1:]
        for line in lines:
            args = [a.strip() for a in line.split(",")]
            result.items.append(
                (args[0].lower(), float(args[1]), float(args[2]), float(args[3]))
            )  # dtcode, van, tot, onderhoudsdiepte
        return result

    def get(self, dtcode: str, chainage: float) -> Optional[float]:
        for i in self.items:
            if i[0] == dtcode.lower() and i[1] <= chainage and chainage <= i[2]:
                return i[3]

        return None
