import pytest

import lotus
from lotus.models import LM


class BaseTest:
    @pytest.fixture(autouse=True)
    def setup(self):
        # Set up any common configurations or fixtures
        lm = LM(model="gpt-4o-mini")
        lotus.settings.configure(lm=lm)
        yield
        # Teardown (if needed)

    # Add any common utility methods here
