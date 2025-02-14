from cyclonedds.qos import Qos, Policy
from cyclonedds.util import duration
from sympy.physics.units import millisecond, milli

class Config:
    H_d405: int=848
    W_d405: int=480

    H_zed: int = 720
    W_zed: int = 1280

class CYCLONE_DEFAULTS:
    QOS = Qos(
        Policy.Reliability.BestEffort,
        Policy.History.KeepLast(1),
    )
    QOS_RPC = Qos(
        Policy.Reliability.Reliable(max_blocking_time=60),
        Policy.Deadline(duration(milliseconds=1000)),
        Policy.History.KeepLast(1),
        Policy.ResourceLimits(
            max_samples=1, max_instances=1, max_samples_per_instance=1
        ),
    )

    QOS_RPC_STATUS = Qos(
        Policy.Reliability.Reliable(max_blocking_time=1),
        Policy.Deadline(duration(milliseconds=1000)),
        Policy.History.KeepLast(1),
        Policy.ResourceLimits(
            max_samples=1, max_instances=1, max_samples_per_instance=1
        ),
    )
