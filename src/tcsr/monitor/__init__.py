from tcsr.monitor.classifiers import LogRegMonitor, RFMonitor, XGBMonitor

__all__ = ["LogRegMonitor", "RFMonitor", "XGBMonitor", "GRUMonitor"]


def __getattr__(name: str):
    if name == "GRUMonitor":
        from tcsr.monitor.temporal_nn import GRUMonitor
        return GRUMonitor
    raise AttributeError(name)
