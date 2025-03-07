# -*- mode: python ; coding: utf-8 -*-
import sys
import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# Get platform details
is_macos = sys.platform == 'darwin'
is_windows = sys.platform == 'win32'
is_linux = sys.platform.startswith('linux')

# Collect hidden imports
hidden_imports = [
    'torch',
    'diffusers',
    'transformers',
    'accelerate',
    'safetensors',
    'PIL',
    'huggingface_hub',
    'ollama',
]

# Additional data files - collect key directories
datas = [
    ('models', 'models'),
    ('docs', 'docs')
]

# Add platform-specific resources
if is_macos and os.path.exists('platform_specific/macos'):
    datas.append(('platform_specific/macos', 'platform_specific/macos'))
elif is_windows and os.path.exists('platform_specific/windows'):
    datas.append(('platform_specific/windows', 'platform_specific/windows'))
elif is_linux and os.path.exists('platform_specific/linux'):
    datas.append(('platform_specific/linux', 'platform_specific/linux'))

# Automatically collect data files from key packages
for pkg in ['transformers', 'diffusers', 'huggingface_hub']:
    datas.extend(collect_data_files(pkg))

a = Analysis(
    ['dream_pixel_forge.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hidden_imports + collect_submodules('diffusers') + collect_submodules('transformers'),
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='DreamPixelForge',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='docs/images/app_icon.icns' if is_macos and os.path.exists('docs/images/app_icon.icns') else 
         'docs/images/app_icon.ico' if is_windows and os.path.exists('docs/images/app_icon.ico') else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='DreamPixelForge',
)

# macOS specific app bundling
if is_macos:
    app = BUNDLE(
        coll,
        name='DreamPixelForge.app',
        icon='docs/images/app_icon.icns' if os.path.exists('docs/images/app_icon.icns') else None,
        bundle_identifier='com.dreamPixelForge',
        info_plist={
            'NSHighResolutionCapable': 'True',
            'NSRequiresAquaSystemAppearance': 'False',  # Enables Dark Mode support
            'CFBundleShortVersionString': '1.0.0',
            'CFBundleDisplayName': 'DreamPixelForge',
            'CFBundleName': 'DreamPixelForge',
        },
    ) 