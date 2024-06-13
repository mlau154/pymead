r"""
Release workflow (ADMINS ONLY):
===============================

On base platform (can be Windows or Linux):
1. Create a draft release on GitHub and enter the version notes
2. Run tests using ``tox run`` in the terminal
3. Make sure version has been updated
4. Make sure all changes are committed and pushed to dev branch
5. Merge with master branch
6. Make a test upload to TestPyPi using the instructions below and test.
7. Generate the standalone executable/installer and copy to the GitHub draft release according to the instructions
   below (if starting from Windows, the installation order should be Windows-->Linux-->macOS; if starting from Linux,
   the installation order should be Linux-->macOS-->Windows)
8. If the test passes, execute the production upload to PyPi. The PyPi upload should come last in the process (right
   before the release is published) because any upload to PyPi cannot be undone.
9. Publish the release on GitHub!

Standalone executable/installer installation instructions:
==========================================================

The standalone files should be built in Python 3.12 to take advantage of the performance improvements.

Windows
-------
- First, navigate to the base directory of *pymead* (the directory containing ``pyproject.toml``)
- Create a fresh virtual environment using PowerShell: ``py -3.12 -m venv .\pymead312``
- May need to run the command `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` the first time
  if the previous command errors out
- Run ``.\pymead312\Scripts\Activate.ps1`` to activate the virtual environment
- Install *pymead* in the new environment: ``pip install .``
- Install *pyinstaller* in the new environment: ``pip install pyinstaller``
- Navigate to the installation directory: ``cd install``
- Generate the installation wizard: ``python -m install``
- Test the new installation. If it passes,
- Remove the virtual environment by navigating back to the base directory and running ``deactivate``, then
  ``rm -r pymead312`` to remove the virtual environment.
- Copy the generated installation wizard to the draft release on GitHub.

Linux
-----
- First, make sure to ``git pull`` to use the latest updates to *pymead*
- Navigate to the base directory of *pymead* (the directory containing ``pyproject.toml``)
- Create a fresh virtual environment: ``python3.12 -m venv pymead312``
- Run ``source ./bin/activate`` to activate the virtual environment
- Install *pymead* in the new environment: ``pip install .``
- Install *pyinstaller* in the new environment: ``pip install pyinstaller``
- Navigate to the installation directory: ``cd install``
- Generate the installation wizard: ``python -m install``
- Test the new installation. If it passes,
- Remove the virtual environment by navigating back to the base directory and running ``deactivate``, then
  ``rm -r pymead312`` to remove the virtual environment.
- Copy the output tarball to the draft release on GitHub.

macOS
-----
From the Linux environment, open the macOS virtual machine by running ``./basic.sh`` from ``~/KVM/macOS.``
Then, follow every step from the Linux instructions exactly.

For uploading to TestPyPi (test) or PyPi (prod):
================================================
- Use ``py -3.10 -m build`` from the root directory to build
- (Prod & Test) Use ``twine check dist/*`` to check the distribution
- (Test Only) Use ``twine upload --repository testpypi dist/*`` to upload to TestPyPi
- (Prod Only) Use ``twine upload dist/*`` to upload to PyPi
- (Test Only) In a fresh, virtual environment, use
  ``pip install --extra-index-url https://test.pypi.org/simple/ pymead==<short ver name>`` to test the installation

To test the TestPyPi build in a fresh virtual environment
=========================================================
First, navigate to any directory outside the base directory. Then, for Python 3.12,
- ``py -3.12 -m venv .\pymead312`` (Windows Powershell) or ``python3.12 -m venv pymead312`` (Linux)
- May need to run the command ``Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`` the first time
  if the previous command errors out on Windows Powershell
- ``cd pymead312``
- ``.\Scripts\Activate.ps1`` (Windows Powershell) or ``source ./bin/activate`` (Linux)
- ``pip install --extra-index-url https://test.pypi.org/simple/ pymead==2.0.0-b5`` to test pymead 2.0.0-beta.5 in a
  Python 3.12 environment
- Once done with the virtual environment, first deactivate using the command ``deactivate`` from any directory. Then,
  navigate to the parent directory of the virtual environment (in this case, ``cd ..``) and ``rm -r pymead312``

For more information on virtual environments, see https://docs.python.org/3/library/venv.html
"""

import os
import platform
import shutil
import subprocess

from pymead.version import __version__


def check_dependencies():
    if shutil.which("pyinstaller") is None:
        raise ValueError("pyinstaller not found on system path. Must install before proceeding.")
    if platform.system() == "Windows" and shutil.which("iscc") is None:
        raise ValueError("iscc Inno Setup command line installer not found on system path. Install and/or add to path"
                         "before proceeding.")


def write_iss_file(iss_file: str, version: str, app_name: str, logo_path: str, output_dir: str, pymead_exe_dir: str):
    lines = [
        '#define MyAppName "pymead"',
        f'#define MyAppVersion "{version}"',
        '#define MyAppPublisher "Matthew Lauer"',
        f'#define MyAppURL "https://pymead.readthedocs.io/"',
        f'#define MyAppExeName "{app_name}"',
        '#define MyAppAssocName MyAppName + " File"',
        '#define MyAppAssocExt ".jmea"',
        '#define MyAppAssocKey StringChange(MyAppAssocName, " ", "") + MyAppAssocExt',
        f'#define MyIconFilename "{logo_path}"',
        "",
        f"[Setup]",
        '; NOTE: The value of AppId uniquely identifies this application. Do not use the same AppId value in installers for other applications.',
        '; (To generate a new GUID, click Tools | Generate GUID inside the IDE.)',
        'AppId={{252E71D6-04D9-4462-83EB-95D46F57883F}',
        'AppName={#MyAppName}',
        'AppVersion={#MyAppVersion}',
        ';AppVerName={#MyAppName} {#MyAppVersion}',
        'AppPublisher={#MyAppPublisher}',
        'AppPublisherURL={#MyAppURL}',
        'AppSupportURL={#MyAppURL}',
        'AppUpdatesURL={#MyAppURL}',
        'DefaultDirName={autopf}\\{#MyAppName}',
        'ChangesAssociations=yes',
        'DisableProgramGroupPage=yes',
        '; Uncomment the following line to run in non administrative install mode (install for current user only.)',
        ';PrivilegesRequired=lowest',
        'OutputBaseFilename=pymeadsetup_{#MyAppVersion}',
        f'OutputDir={output_dir}',
        'Compression=lzma',
        'SetupIconFile={#MyIconFilename}',
        'SolidCompression=yes',
        'WizardStyle=modern',
        "",
        "[Languages]",
        'Name: "english"; MessagesFile: "compiler:Default.isl"',
        "",
        "[Tasks]",
        'Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked',
        "",
        "[Files]",
        fr'Source: "{pymead_exe_dir}\*"; DestDir: "{{app}}"; Flags: ignoreversion recursesubdirs createallsubdirs',
        '; NOTE: Don\'t use "Flags: ignoreversion" on any shared system files',
        "",
        "[Registry]",
        r'Root: HKA; Subkey: "Software\Classes\{#MyAppAssocExt}\OpenWithProgids"; ValueType: string; ValueName: "{#MyAppAssocKey}"; ValueData: ""; Flags: uninsdeletevalue',
        r'Root: HKA; Subkey: "Software\Classes\{#MyAppAssocKey}"; ValueType: string; ValueName: ""; ValueData: "{#MyAppAssocName}"; Flags: uninsdeletekey',
        r'Root: HKA; Subkey: "Software\Classes\{#MyAppAssocKey}\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\{#MyAppExeName},0"',
        r'Root: HKA; Subkey: "Software\Classes\{#MyAppAssocKey}\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#MyAppExeName}"" ""%1"""',
        r'Root: HKA; Subkey: "Software\Classes\Applications\{#MyAppExeName}\SupportedTypes"; ValueType: string; ValueName: ".myp"; ValueData: ""',
        "",
        "[Icons]",
        r'Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"',
        r'Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon',
        "",
        "[Run]",
        'Filename: "{app}\\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, \'&\', \'&&\')}}"; Flags: nowait postinstall skipifsilent',
        ""
    ]
    with open(iss_file, "w") as iss:
        iss.write("\n".join(lines))


def run(create_installer: bool = True):
    # Check dependencies
    check_dependencies()

    # Get current OS
    system = platform.system()

    # Set directories
    top_level_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    install_dir = os.path.join(top_level_dir, "install")
    build_dir = os.path.join(install_dir, "build")
    dist_dir = os.path.join(install_dir, "dist")
    gui_dir = os.path.join(top_level_dir, "pymead", "gui")
    exe_dir = os.path.join(dist_dir, "pymead")
    exe_name = "pymead.exe" if system == "Windows" else "pymead"
    iss_file = os.path.join(install_dir, "pymead_installer.iss")

    print("Running pyinstaller to compile pymead as a frozen executable...")
    pyinstaller_process = subprocess.run([
        "pyinstaller", "pymead.spec",
        "--workpath", build_dir,
        "--distpath", dist_dir,
        "--noconfirm"
    ], cwd=gui_dir)

    if pyinstaller_process.returncode == 0:
        pass
    else:
        raise ValueError("pyinstaller command failed")

    if system == "Windows":
        print("Copying executable...")
        shutil.copy(os.path.join(exe_dir, exe_name), install_dir)

        if not create_installer:
            return

        # Run the installer compiler
        print("Writing ISS file...")
        write_iss_file(iss_file=iss_file,
                       version=__version__,
                       app_name=exe_name,
                       logo_path=os.path.join(top_level_dir, "pymead", "icons", "pymead-logo.ico"),
                       output_dir=install_dir,
                       pymead_exe_dir=exe_dir
                       )

        print("Running Inno Setup to compile the setup wizard...")
        iss_process = subprocess.run(["iscc", iss_file])

        if iss_process.returncode == 0:
            pass
        else:
            raise ValueError("iss setup install command failed")

    elif system in ["Linux", "Darwin"]:  # Note: "Darwin" is the platform system name for macOS
        print("Compressing app...")
        platform_descriptor = "linux" if system == "Linux" else "macOS"
        tarball_name = f"pymead-{__version__}-{platform_descriptor}.tar.gz"
        tarball_process = subprocess.run(["tar", "-czvf", tarball_name, "pymead"], cwd=dist_dir)

        if tarball_process.returncode == 0:
            pass
        else:
            raise ValueError("Tarball compression command failed")

        print("Moving tarball to install directory...")
        shutil.move(os.path.join(dist_dir, tarball_name), install_dir)

    print(f"Install complete. Output is in {install_dir}")


if __name__ == "__main__":
    run(create_installer=True)
