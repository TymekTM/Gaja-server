import os, sys, importlib
# Ensure repository root & server directory are on path before shims
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
SERVER = os.path.abspath(os.path.join(ROOT, 'Gaja-server'))
for p in (ROOT, SERVER):
	if p not in sys.path:
		sys.path.insert(0, p)

# Force using real config modules (skip stub for performance/feature parity)
os.environ.setdefault('GAJA_USE_REAL_CONFIG','1')

# Try importing real config package early
try:  # pragma: no cover
	importlib.import_module('config.config_loader')
	importlib.import_module('config.config_manager')
except Exception:
	pass

from compat.legacy_shims import install_all  # noqa: E402

# Install legacy shims only after attempting real imports, to avoid stub shadowing.
install_all()

# Environment toggles for faster / isolated tests
os.environ.setdefault('GAJA_LATENCY_TRACE','0')  # disable tracer for speed if not explicitly enabled
