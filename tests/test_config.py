#!/usr/bin/env python3
# Copyright (c) 2026 Kishore Sridhar - 611451003
# Tatung University — I4210 AI實務專題

import dms.config as config

def test_config_variables():
    """Verify the core threshold variables are loaded correctly."""
    # We can now access the variables directly from the config module
    assert config.CAMERA_WIDTH == 640
    assert config.CAMERA_HEIGHT == 320
    assert config.EAR_THRESHOLD == 0.20
    assert not config.ALERT_MOCK