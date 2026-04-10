"""
Pages 模組初始化
"""
from .page1_data_collection import Page1DataCollection
from .page2_model_management import Page2ModelManagement
from .page3_recognition import Page3Recognition
from .page4_body_tracking import Page4BodyTracking

__all__ = [
    'Page1DataCollection',
    'Page2ModelManagement',
    'Page3Recognition',
    'Page4BodyTracking'
]
