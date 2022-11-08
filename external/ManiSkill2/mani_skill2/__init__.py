from pathlib import Path

ROOT_DIR = Path(__file__).parent.resolve()
ASSET_DIR = ROOT_DIR / "assets"
AGENT_CONFIG_DIR = ASSET_DIR / "config_files/agents"
DESCRIPTION_DIR = ASSET_DIR / "descriptions"


def get_commit_info(with_modified_files=False, with_untracked_files=False):
    import pprint

    import git

    try:
        repo = git.Repo(ROOT_DIR, search_parent_directories=True)
    except git.InvalidGitRepositoryError as err:
        return "mani_skill2 is not installed with git."
    else:
        commit_id = repo.head.commit
        info_str = f"commit_id: {commit_id}"
        
        if with_modified_files:
            # https://stackoverflow.com/questions/33733453/get-changed-files-using-gitpython
            modified_files = [item.a_path for item in repo.index.diff(None)]
            info_str += '\n' + "Modified files:" + pprint.pformat(modified_files)
        
        if with_untracked_files:
            untracked_files = repo.untracked_files
            info_str += '\n' + "Untracked files:" + pprint.pformat(untracked_files)
        
        # https://github.com/gitpython-developers/GitPython/issues/718#issuecomment-360267779
        repo.__del__()
        return info_str
