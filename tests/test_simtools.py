
import pytest
import numpy as np

from ami.simtools import Status, Evaluator


# simtools.Status ------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("num, start, sta_update",
                         [
                             (10, 0, 1),
                             (10, 'a', 'b'),
                             (100, 2.2, 4.4),
                             (1000, {'value': 'untested'}, {'value': 'updated'}),
                             (10, None, 1)  # Can initialise with None
                         ])
def test_status_pass(num, start, sta_update):
    """Tests the initialisation and all methods.
    """

    # initialisation
    st = Status(num, start)
    assert len(st.state) == num
    assert sum(st.state == start) == num
    assert len(st.changelog) == 0
    assert len(st.untested()) == num
    assert len(st.tested()) == 0

    # update, tested, untested
    indices = np.random.choice(num, size=3, replace=False)
    one, multiple = indices[0], indices[1:]

    # single update passed
    st.update(one, sta_update)
    num_updated = 1
    assert sum(st.state == sta_update) == num_updated
    assert len(st.tested()) == num_updated
    assert sum(st.state != sta_update) == (num - num_updated)
    assert len(st.untested()) == (num - num_updated)

    # multiple update passed
    st.update(multiple, sta_update)
    num_updated += 2
    assert sum(st.state == sta_update) == num_updated
    assert len(st.tested()) == num_updated
    assert sum(st.state != sta_update) == (num - num_updated)
    assert len(st.untested()) == (num - num_updated)


@pytest.mark.xfail(reason='Invalid inputs and misuse of API')
@pytest.mark.parametrize("num, start, sta_update",
                         [
                             (0, 0, 1),  # positive int to initalise
                             (10, 0, 2.5),  # updating different numerical types
                             (10, 'a', 'bc'),  # updating different string lengths
                             (10, 1, None)  # cannot update with None
                         ])
def test_status_fail(num, start, sta_update):
    st = Status(num, start)
    st.update(0, sta_update)

    assert sum(st.state == sta_update) == 1
    assert len(st.tested()) == 1
    assert sum(st.state != sta_update) == (num - 1)
    assert len(st.untested()) == (num - 1)


# simtools.Evaluator ---------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("top_ind, y, n, found, expected_top",
                         [
                             (np.arange(95, 100),
                              np.arange(0, 100)*3+4,
                              5,
                              [95, 96, 1, 5, 3, 56],
                              [95, 96])
                         ])
def test_evaluator_pass(top_ind, y, n, found, expected_top):
    """Shouldn't expect duplicate counts of indices when comparing found and top values.
    """
    ev = Evaluator(top_ind)
    ev_f = Evaluator.from_unordered(y, n)
    ev_top = np.sort(ev.get_top_n())
    ev_f_top = np.sort(ev_f.get_top_n())

    assert len(ev) == len(top_ind)  # check lengths match.
    assert np.array_equal(top_ind, ev_top)  # check initialised well and method works correctly.
    assert np.array_equal(ev_top, ev_f_top)  # check factory method doesn't alter values

    are_top = np.sort(ev.top_found_id(found))
    are_top_count = ev.top_found_count(found)

    assert len(are_top) == are_top_count  # Values should match as fed by same private function.
    assert np.array_equal(are_top, expected_top)  # get expected values out


def test_evaluator_duplicate_pass():
    """Checks duplicate values aren't duplicated when considering count of top values deterined.
    """
    top = [1]
    ev = Evaluator(top)
    found = [1, 1, 2]
    assert ev.top_found_count(found) == 1


@pytest.mark.xfail(reason='Ensures duplicate values arent considered multiple times')
def test_evaluator_duplicate_fail():
    top = [1]
    ev = Evaluator(top)
    found = [1, 1, 2]
    assert np.array_equal(ev.top_found_count(found), [1, 1])


# ----------------------------------------------------------------------------------------------------------------------
