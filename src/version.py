from importlib.metadata import PackageNotFoundError, version


try:
    APP_VERSION = version("smartroute-ai")
except PackageNotFoundError:
    APP_VERSION = "2.1.0"
