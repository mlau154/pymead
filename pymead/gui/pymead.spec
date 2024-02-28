# -*- mode: python ; coding: utf-8 -*-


block_cipher = None


a = Analysis(
    ['gui.py'],
    pathex=[],
    binaries=[],
    datas=[('gui_settings/defaults/*.json', 'pymead/gui/gui_settings/defaults/'),
    ('gui_settings/themes/*.json', 'pymead/gui/gui_settings/themes/'),
    ('gui_settings/*.json', 'pymead/gui/gui_settings/'),
    ('default_airfoil/*.jmea', 'pymead/gui/default_airfoil/'),
    ('../icons/*.png', 'pymead/icons/'),
    ('../icons/*.ico', 'pymead/icons/'),
    ('../icons/*.svg', 'pymead/icons/'),
    ('dialog_widgets/*.json', 'pymead/gui/dialog_widgets/'),
    ('../resources', 'pymead/resources')],
    hiddenimports=[],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

splash = Splash('../icons/pymead-splash.png',
                binaries=a.binaries,
                datas=a.datas,
                text_pos=(40, 200),
                text_size=14,
                text_color='black')

exe = EXE(
    pyz,
    splash,
    a.scripts,
    [],
    exclude_binaries=True,
    name='pymead',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    splash='../icons/pymead-logo.png',
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='../icons/pymead-logo.ico',
)
coll = COLLECT(
    exe,
    splash.binaries,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='pymead',
)
