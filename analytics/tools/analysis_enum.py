import enum


class GreenHouseType(enum.Enum):
    Phenotyping = "PHENOTYPING"
    Commercial = "COMMERCIAL"


class FileFormats(enum.Enum):
    WRONG = -1
    MANUAL = "MANUAL"
    NEW = "NEW"
    RAW = "RAW"
    OFFLINE = "OFFLINE"


class FileTypes(enum.Enum):
    WRONG = -1
    CSV = "CSV"
    FEATHER = "FEATHER"


class AnalysisMode(enum.Enum):
    WRONG = -1
    AGGREGATE = "AGGREGATE"
    REPLACE = "REPLACE"
    REFRESH = "REFRESH"


class FruitTypes(enum.Enum):
    POMEGRANATE = 'Pomegranate',
    MANDARIN = 'Mandarin',
    PEACH = 'Peach',
    NECTARINE = 'Nectarine',
    LEMON = 'Lemon',
    ORANGE = 'Orange',
    APPLE = 'Apple',
    PEAR = 'Pear',
    LIME = 'Lime',
    PLUM = 'Plum',
    WINE_GRAPES = 'Wine Grapes',
    TABLE_GRAPES = 'Table Grapes',
    MANGO = 'Mango',
    CHERRY = 'Cherry',
    GRAPEFRUIT = 'Grapefruit',
    POMELO = 'Pomelo',
    APRICOT = 'Apricot',
    PERSIMMON = 'Persimmon',
    KIWI = 'Kiwi',
    WALNUT = 'Walnut',
    TOMATO = 'Tomato',
    PEPPER = 'Pepper'

