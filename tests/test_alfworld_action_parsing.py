import importlib.util
from pathlib import Path

import pytest

from skillopt.envs.alfworld.actions import extract_action
from skillopt.envs.alfworld.rollout import _extract_action


def _load_projection():
    # Load the module file directly: the vendor package __init__ imports the
    # alfworld simulator, which CI does not install.
    import skillopt.envs.alfworld as alfworld_pkg

    path = Path(alfworld_pkg.__file__).parent / "vendor" / "alfworld_projection.py"
    spec = importlib.util.spec_from_file_location("_alfworld_projection_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.alfworld_projection


alfworld_projection = _load_projection()


CLOSING_TAG_IN_THINK = (
    "<think>I must end with a </action> tag. Go to the desk.</think>"
    "<action>go to desk 1</action>"
)
FORMAT_EXAMPLE_IN_THINK = (
    "<think>The format is <action>look</action>. I will open the drawer.</think>"
    "<action>open drawer 1</action>"
)
UPPERCASE_TAGS = "<think>ok</think><Action>go to desk 1</Action>"


@pytest.mark.parametrize(
    "response, expected",
    [
        ("<think>ok</think><action>go to desk 1</action>", "go to desk 1"),
        (CLOSING_TAG_IN_THINK, "go to desk 1"),
        (FORMAT_EXAMPLE_IN_THINK, "open drawer 1"),
        (UPPERCASE_TAGS, "go to desk 1"),
        ("<think>ok</think><action>\n  take apple 1 from table 1 \n</action>", "take apple 1 from table 1"),
        ("<think>no tag</think>go to desk 1", None),
        ("<think>open only</think><action>go to desk 1", None),
        ("", None),
    ],
)
def test_extract_action(response, expected):
    assert extract_action(response) == expected
    assert _extract_action(response) == expected


@pytest.mark.parametrize("response", [CLOSING_TAG_IN_THINK, FORMAT_EXAMPLE_IN_THINK, UPPERCASE_TAGS])
def test_projection_executes_the_recorded_action(response):
    actions, valids = alfworld_projection([response], [[]])

    assert actions == [_extract_action(response).lower()]
    assert valids == [1]


def test_projection_closing_tag_in_think_is_not_an_empty_action():
    actions, valids = alfworld_projection([CLOSING_TAG_IN_THINK], [[]])

    assert actions == ["go to desk 1"]
    assert valids == [1]


def test_projection_keeps_existing_validity_rules():
    no_action = "<think>thinking</think> I will go to the desk"
    no_think = "<action>go to desk 1</action>"
    chinese = "<think>去桌子</think><action>go to desk 1</action>"

    actions, valids = alfworld_projection([no_action, no_think, chinese], [[], [], []])

    assert actions[0] == no_action.lower()[-30:]
    assert actions[1:] == ["go to desk 1", "go to desk 1"]
    assert valids == [0, 0, 0]
