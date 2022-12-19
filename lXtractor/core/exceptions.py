class InitError(ValueError):
    """
    A broad category exception for problems with
    an object's initialization
    """

    pass


class MissingData(ValueError):
    pass


class AmbiguousData(ValueError):
    pass


class AmbiguousMapping(ValueError):
    pass


class NoOverlap(ValueError):
    pass


class FormatError(ValueError):
    pass


class FailedCalculation(RuntimeError):
    pass


class LengthMismatch(ValueError):
    pass


class OverlapError(ValueError):
    pass


class ParsingError(ValueError):
    pass


if __name__ == '__main__':
    raise RuntimeError
