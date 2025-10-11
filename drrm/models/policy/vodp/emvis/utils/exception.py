class CustomException(Exception):
    def __str__(self):
        return "This is a custom exception."