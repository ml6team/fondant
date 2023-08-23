# from .pa_head import PA_Head
from .fast_head import fast_head

# from .psenet_head import PSENet_Head
from .builder import build_head

# __all__ = ['PA_Head', 'fast_head', 'PSENet_Head',
#            'build_head']

__all__ = ["fast_head", "build_head"]
