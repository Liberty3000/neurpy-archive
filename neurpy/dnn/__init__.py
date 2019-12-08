from . import classifier
from . import architectures
from . import weight_init
from . import metrics

classifiers = {'darknet53':lambda *args,**kwargs:architectures.darknet.darknet53(*args,**kwargs),
                'resnet18':lambda *args,**kwargs:architectures.resnet.resnet50(*args,**kwargs)}
