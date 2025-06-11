from enum import StrEnum

class ExpenseCategory(StrEnum):
    MEALS = "meals"
    TRAVEL = "travel"
    LODGING = "lodging"
    ENTERTAINMENT = "entertainment"
    TRAINING = "training"
    GIFTS = "gifts"
    EDUCATION = "education"
    OFFICE_SUPPLIES = "office_supplies"
    OTHER = "other"