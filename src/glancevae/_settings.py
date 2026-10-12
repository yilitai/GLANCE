import logging
import sys
from pathlib import Path
from typing import Literal, Union

from tqdm import tqdm as _tqdm


class _TqdmHandler(logging.StreamHandler):
    """
    A StreamHandler that routes log records through ``tqdm.write``
    so that log messages do not interleave with or add blank lines
    between active tqdm progress bars.
    """

    def emit(self, record: logging.LogRecord) -> None:
        try:
            _tqdm.write(self.format(record))
        except Exception:
            self.handleError(record)


# Module-level logger for the entire glancevae package.
# All submodules obtain their logger via logging.getLogger("glancevae"),
# which is a child of this root logger and inherits its handlers.
_logger = logging.getLogger("glancevae")
_logger.setLevel(logging.DEBUG)  # Logger itself is always permissive;
# handlers control what actually appears.


def _make_handler(
    verbosity: Literal["print", "file", "off"],
    log_file: Union[str, Path, None],
) -> logging.Handler:
    """Build a single handler that matches the requested verbosity mode."""
    if verbosity == "off":
        return logging.NullHandler()

    formatter = logging.Formatter("%(message)s")

    if verbosity == "print":
        handler = _TqdmHandler(sys.stdout)
    else:  # "file"
        path = Path(log_file) if log_file is not None else Path("glance.log")
        handler = logging.FileHandler(path, mode="a", encoding="utf-8")

    handler.setFormatter(formatter)
    return handler


class GLANCESettings:
    """
    Global settings for the glancevae package.

    Attributes
    ----------
    verbosity : {"print", "file", "off"}
        Controls where log output is sent.

        ``"print"``
            Default. All messages go to stdout, identical to ``print``
            behaviour.  tqdm progress bars are shown.
        ``"file"``
            Messages are written to ``log_file`` only; nothing appears
            in the terminal.  tqdm progress bars are suppressed.
        ``"off"``
            All output is suppressed, including tqdm progress bars.

    log_file : str, Path, or None
        Destination file used when ``verbosity="file"``.  Relative paths
        are resolved against the current working directory at the time
        ``verbosity`` is set to ``"file"``.  Defaults to ``"glance.log"``
        in the working directory.  Ignored when verbosity is not
        ``"file"``.

    Examples
    --------
    Default (print to terminal)::

        from glancevae import settings
        model.fit_transform(...)   # progress bar + log messages visible

    Write to file instead::

        from glancevae import settings
        settings.log_file = "results/glance.log"
        settings.verbosity = "file"

    Suppress all output (e.g. in a pipeline script)::

        from glancevae import settings
        settings.verbosity = "off"
    """

    def __init__(self):
        self._verbosity: Literal["print", "file", "off"] = "print"
        self._log_file: Union[str, Path, None] = None
        self._apply()

    # ------------------------------------------------------------------
    # verbosity
    # ------------------------------------------------------------------

    @property
    def verbosity(self) -> Literal["print", "file", "off"]:
        return self._verbosity

    @verbosity.setter
    def verbosity(self, value: Literal["print", "file", "off"]):
        if value not in ("print", "file", "off"):
            raise ValueError(
                f"verbosity must be one of 'print', 'file', or 'off'; got {value!r}"
            )
        self._verbosity = value
        self._apply()

    # ------------------------------------------------------------------
    # log_file
    # ------------------------------------------------------------------

    @property
    def log_file(self) -> Union[str, Path, None]:
        return self._log_file

    @log_file.setter
    def log_file(self, value: Union[str, Path, None]):
        self._log_file = value
        # Re-apply only if we are already in file mode so the new path
        # takes effect immediately.
        if self._verbosity == "file":
            self._apply()

    # ------------------------------------------------------------------
    # tqdm helper
    # ------------------------------------------------------------------

    @property
    def tqdm_disabled(self) -> bool:
        """True when tqdm progress bars should be suppressed."""
        return self._verbosity in ("file", "off")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _apply(self):
        """Replace all handlers on the root glancevae logger."""
        _logger.handlers.clear()
        _logger.addHandler(_make_handler(self._verbosity, self._log_file))

    def __repr__(self) -> str:
        return (
            f"GLANCESettings(verbosity={self._verbosity!r}, "
            f"log_file={self._log_file!r})"
        )


# Singleton exposed via ``from glancevae import settings``.
settings = GLANCESettings()
