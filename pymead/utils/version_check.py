import requests
from dataclasses import dataclass

from pymead.version import __version__


def get_current_pymead_version():
    """
    Gets the current pymead version from the top-level __version__

    Returns
    =======
    str
        Current version of pymead in raw string form
    """
    return "v" + __version__


def get_latest_pymead_release_version():
    """
    Gets the latest raw version string from pymead's GitHub release page (note that the order of the list generated
    by `response.json()` is in order from latest to oldest.

    Returns
    =======
    str
        Raw version string representing the latest version of pymead
    """
    response = requests.get("https://api.github.com/repos/mlau154/pymead/releases")
    latest_version_string = response.json()[0]["tag_name"]
    return latest_version_string


@dataclass
class VersionInfo:
    major: int
    minor: int
    patch: int
    pre_release_tag: str or None
    pre_release_id: int or None


def extract_version_info_from_string(version_string: str):
    """
    Takes a raw string of the form 'v[major].[minor].[patch]-[pre-release tag].[pre-release id]' and
    returns a namedtuple containing the fields identified in brackets.

    Parameters
    ==========
    version_string: str
        Raw version string

    Returns
    =======
    VersionInfo
    """
    v_split = version_string.split("-")

    if "v" in v_split[0]:
        major_minor_patch_str = v_split[0][1:]
    else:
        major_minor_patch_str = v_split[0][:]

    major = int(major_minor_patch_str.split(".")[0])
    minor = int(major_minor_patch_str.split(".")[1])
    patch = int(major_minor_patch_str.split(".")[2])

    pre_release_tag = None
    pre_release_id = None

    if len(v_split) > 1:
        pre_release_tag = v_split[1].split(".")[0]
        pre_release_id = int(v_split[1].split(".")[1])

    return VersionInfo(major=major, minor=minor, patch=patch,
                       pre_release_tag=pre_release_tag, pre_release_id=pre_release_id)


def compare_versions(version_1: VersionInfo or str, version_2: VersionInfo or str):
    """
    Compares two versions. If the two versions are equal, return 2. If version 2 is newer than version 1, return 1.
    If version 2 is older than version 1, return 0.

    Parameters
    ==========
    version_1: VersionInfo or str
        First version to compare

    version_2: VersionInfo or str
        Second version to compare

    Returns
    =======
    int
        Flag representing the result of the version comparison. 0: version 2 is older than version 1. 1: version 2
        is newer than version 1. 2: the versions match.
    """

    if not isinstance(version_1, VersionInfo):
        version_1 = extract_version_info_from_string(version_1)
    if not isinstance(version_2, VersionInfo):
        version_2 = extract_version_info_from_string(version_2)

    def compare_values(v1: int, v2: int):
        if v2 == v1:
            return None
        elif v2 > v1:
            return 1
        else:
            return 0

    # First, compare the major version
    major_comparison = compare_values(version_1.major, version_2.major)
    if major_comparison is not None:
        return major_comparison

    # Next, compare the minor version
    minor_comparison = compare_values(version_1.minor, version_2.minor)
    if minor_comparison is not None:
        return minor_comparison

    # Then, compare the patch version
    patch_comparison = compare_values(version_1.patch, version_2.patch)
    if patch_comparison is not None:
        return patch_comparison

    # If the major, minor, and patch versions all match but only one version has a pre-release tag, use this information
    # directly to determine the result of the comparison
    if version_1.pre_release_tag is None and version_2.pre_release_tag is not None:
        return 0
    if version_1.pre_release_tag is not None and version_2.pre_release_tag is None:
        return 1

    # If the pre-release tags do not match, use this information directly to determine the result of the comparison
    if version_1.pre_release_tag != version_2.pre_release_tag:
        if version_1.pre_release_tag == "alpha" and version_2.pre_release_tag == "beta":
            return 1
        elif version_1.pre_release_tag == "beta" and version_2.pre_release_tag == "alpha":
            return 0
        else:
            raise ValueError(f"Found invalid pre_release_tag {version_1.pre_release_tag = }, "
                             f"{version_2.pre_release_tag = }. Valid pre-release tag names are 'alpha' and 'beta'.")

    # Perform the pre-release id comparison
    pre_release_id_comparison = compare_values(version_1.pre_release_id, version_2.pre_release_id)
    if pre_release_id_comparison is not None:
        return pre_release_id_comparison

    # If the function has not yet returned a value, this means that the version strings match
    return 2


def using_latest(current_version_string: str or None = None):
    """
    Compares the current version string to the latest version string. If the latest version string is > the current
    version string, return False, else return True. If the current version string is None, pull the version from
    `pymead.version.__version__`.

    Parameters
    ==========
    current_version_string: str or None
        The version of pymead in use. Default: `None`

    Returns
    =======
    typing.Tuple[bool, str, str]
        First element: True if using the latest version, False otherwise. Second element: string representing the latest
        version of pymead. Third element: string representing the current version of pymead.
    """
    if current_version_string is None:
        current_version_string = get_current_pymead_version()

    current_version = extract_version_info_from_string(current_version_string)

    latest_version_string = get_latest_pymead_release_version()
    latest_version = extract_version_info_from_string(latest_version_string)

    return bool(compare_versions(latest_version, current_version)), latest_version_string, current_version_string
