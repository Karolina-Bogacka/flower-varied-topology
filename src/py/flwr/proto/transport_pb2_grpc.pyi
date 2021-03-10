"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file
"""
import abc
import grpc
import typing

from .transport_pb2 import *
class FlowerServiceStub:
    def __init__(self, channel: grpc.Channel) -> None: ...
    def Join(self,
        request: typing.Iterator[global___ClientMessage],
    ) -> typing.Iterator[global___ServerMessage]: ...


class FlowerServiceServicer(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def Join(self,
        request: typing.Iterator[global___ClientMessage],
        context: grpc.ServicerContext,
    ) -> typing.Iterator[global___ServerMessage]: ...


def add_FlowerServiceServicer_to_server(servicer: FlowerServiceServicer, server: grpc.Server) -> None: ...