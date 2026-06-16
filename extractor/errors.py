class ExtractorError(RuntimeError):
    """Base error for extractor runtime."""


class ConfigViolation(ExtractorError):
    pass


class DependencyViolation(ExtractorError):
    pass


class PreprocessViolation(ExtractorError):
    pass


class ProviderExecutionViolation(ExtractorError):
    pass


class AdapterViolation(ExtractorError):
    pass


class FactPackViolation(ExtractorError):
    pass


class ContaminationViolation(ExtractorError):
    pass


class HandoffViolation(ExtractorError):
    pass
