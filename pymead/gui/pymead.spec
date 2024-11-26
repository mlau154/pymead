# -*- mode: python ; coding: utf-8 -*-


block_cipher = None


a = Analysis(
    ['gui.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('gui_settings/defaults/*.json', 'pymead/gui/gui_settings/defaults/'),
        ('gui_settings/themes/*.json', 'pymead/gui/gui_settings/themes/'),
        ('gui_settings/*.json', 'pymead/gui/gui_settings/'),
        ('../icons/*.png', 'pymead/icons/'),
        ('../icons/*.ico', 'pymead/icons/'),
        ('../icons/*.svg', 'pymead/icons/'),
        ('dialog_widgets/*.json', 'pymead/gui/dialog_widgets/'),
        ('../resources', 'pymead/resources'),
        ('../tests/core_tests/*.jmea', 'pymead/tests/core_tests/'),
        ('../examples/*.jmea', 'pymead/examples/'),
        ('../tests/misc_tests/ps_svg_pdf_conversion/*.ps', 'pymead/tests/misc_tests/ps_svg_pdf_conversion/'),
        ('../tests/opt_tests/*.json', 'pymead/tests/opt_tests/'),
        ('../tests/opt_tests/*.jmea', 'pymead/tests/opt_tests/'),
        ('../tests/gui_tests/*.txt', 'pymead/tests/gui_tests/'),
        ('../tests/gui_tests/*.json', 'pymead/tests/gui_tests/'),
        ('../tests/gui_tests/data/iso_prop/*', 'pymead/tests/gui_tests/data/iso_prop/'),
        ('../tests/gui_tests/data/iso_prop_test_0/*', 'pymead/tests/gui_tests/data/iso_prop_test_0/'),
        ('../tests/gui_tests/data/analysis_temp_iso_prop/ga_airfoil_3/*', 'pymead/tests/gui_tests/data/analysis_temp_iso_prop/ga_airfoil_3/'),
    ],
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

#splash = Splash('../icons/pymead-splash.png',
                #binaries=a.binaries,
                #datas=a.datas,
                #text_pos=(40, 200),
                #text_size=14,
                #text_color='black')

exe = EXE(
    pyz,
    #splash,
    a.scripts,
    [],
    exclude_binaries=True,
    name='pymead',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    #splash='../icons/pymead-logo.png',
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='../icons/pymead-logo.ico',
)
coll = COLLECT(
    exe,
    #splash.binaries,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='pymead',
)
