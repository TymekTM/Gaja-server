import os
from compat.legacy_shims import install_all

# Install legacy shims before any test imports modules expecting flat names.
install_all()

# Environment toggles for faster / isolated tests
os.environ.setdefault('GAJA_LATENCY_TRACE','0')  # disable tracer for speed if not explicitly enabled
