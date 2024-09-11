# noinspection PyProtectedMember
from .internals.scoped_value_type_dispatch import (
    _ReparametrizeArgument,
    _ReparametrizeArgumentByType,
)

reparametrize_argument_by_value = _ReparametrizeArgument()
reparametrize_argument_by_type = _ReparametrizeArgumentByType()


# The default implementation of reparametrize_argument_by_value, just defers to the type dispatcher, so this can
#  be used as the default conversion mechanism.
# Using wrapper to make this non-registerable, requiring that the other two get used for explicitness.
def reparametrize_argument(*args, **kwargs):
    return reparametrize_argument_by_value(*args, **kwargs)
