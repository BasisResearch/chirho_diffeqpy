from functools import singledispatch
from .singledispatch_on_value_and_test_arg import valuedispatch


# 1. Reparametrization starts with an attempt at value dispatch on a per-test/per-arg basis, which then falls
#  back to a general value dispatch.
@valuedispatch
def reparametrize_argument(dispatch_arg, *args, **kwargs):
    # 2. And if no specific dispatcher is found for the test/arg pair or the value generally,
    #  we fall back to type dispatch, which is handled with _reparametrize_argument_by_type
    try:
        reparametrize_argument_by_type(dispatch_arg, *args, **kwargs)
    except NotImplementedError as e:
        raise NotImplementedError(f"Failed to find dispatch by value and test/arg pair, by just value, or by type"
                                  f" for {dispatch_arg} of type {type(dispatch_arg)}") from e


@singledispatch
def reparametrize_argument_by_type(dispatch_arg, *args, **kwargs):
    raise NotImplementedError(f"Failed to find dispatch by type for {type(dispatch_arg)}.")
