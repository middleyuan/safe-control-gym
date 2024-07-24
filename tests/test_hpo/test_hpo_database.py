import munch
import pytest

from safe_control_gym.hyperparameters.database import create, drop


@pytest.mark.parametrize('ALGO', ['ilqr', 'ppo', 'sac', 'gp_mpc', 'gpmpc_acados'])
def test_hpo_database(ALGO):

    # create database
    create(munch.Munch({'tag': f'{ALGO}_hpo'}))

    # drop database
    drop(munch.Munch({'tag': f'{ALGO}_hpo'}))
