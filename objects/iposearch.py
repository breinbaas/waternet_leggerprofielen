class IPOSearch:
    def __init__(self):
        self.items = []

    @classmethod
    def from_csv(cls, csv_file):
        result = IPOSearch()
        lines = open(csv_file, "r").readlines()
        for line in lines[1:]:
            args = [a.strip() for a in line.split(",")]
            code = args[0]
            start = float(args[1])
            end = float(args[2])
            ipo = args[3]
            result.items.append((code, start, end, ipo))

        return result

    def get_ipo(self, code, chainage):
        for i in self.items:
            if i[0] == code and i[1] <= chainage and chainage <= i[2]:
                return i[3]
        return None
