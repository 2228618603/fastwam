from pathlib import Path
import hashlib
import json
import os
import shutil

root = Path('/home/zzd/project')
audit = root / 'rtc_engineering_optimization_20260917'
src = root / 'robot1_zzd_test_bundle/home_zzd_test/rtc_python'
main = root / 'giga-world-policy'
expected = json.loads((audit / 'main_sync_files.json').read_text())
# First validate all preimages; no partial source replacement on an unexpected diff.
for rel, digest in expected.items():
    target = main / rel
    current = hashlib.sha256(target.read_bytes()).hexdigest() if target.exists() else None
    desired = hashlib.sha256((src / rel).read_bytes()).hexdigest()
    if current not in (digest, desired):
        raise RuntimeError(f'Unexpected remote modification: {target}')
for rel in expected:
    target = main / rel
    backup = audit / 'before/main' / rel
    if target.exists() and not backup.exists():
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target, backup)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + '.rtc_opt_tmp')
    shutil.copy2(src / rel, tmp)
    os.replace(tmp, target)
print(f'Synced {len(expected)} main files, original files backed up.')

package = root / 'wam_local_emptybox_deploy_package'
for name in ['README.md', 'build_python_package.sh', 'gwp_aligned_emptybox_python_deploy.tar.gz',
             'gwp_aligned_emptybox_python_deploy.tar.gz.sha256']:
    source = package / name
    backup = audit / 'before/package' / name
    backup.parent.mkdir(parents=True, exist_ok=True)
    if source.exists() and not backup.exists():
        shutil.copy2(source, backup)

builder = package / 'build_python_package.sh'
text = builder.read_text()
marker = 'cp -a "${OUTPUT_ROOT}/README.md" "${PACKAGE}/PACKAGE_README.md"\n'
addition = '''
# 原包只有 deploy/ 入口，没有 RTC 根目录启动脚本和工程优化说明。
# 保留原打包步骤，并补齐这些文件，避免重建后丢失新入口。
cp -a "${SOURCE_ROOT}/README_RTC_ENGINEERING_OPTIMIZATIONS.md" "${PACKAGE}/"
cp -a "${SOURCE_ROOT}/README_RTC_ENGINEERING_OPTIMIZATIONS.md" "${PACKAGE}/README.md"
for script in "${SOURCE_ROOT}"/start_rtc*.sh "${SOURCE_ROOT}"/start_client_local_log_only.sh; do
    cp -a "${script}" "${PACKAGE}/"
done
'''
if addition not in text:
    assert text.count(marker) == 1
    builder.write_text(text.replace(marker, marker + addition))
readme = package / 'README.md'
notice = '''> 2026-09-17 更新：已加入 aligned RTC 工程优化（完整 compile/prefix cache、同线程预热、客户端缩图）。
> 保留 zzd 的 RTC30k BF16 权重和 aligned 布局。原包和构建脚本已备份到
> `/home/zzd/project/rtc_engineering_optimization_20260917/before/package/`。
> 完整变更和启动说明见 `README_RTC_ENGINEERING_OPTIMIZATIONS.md`。
> 已有机器人 rtc_python 目录建议使用 `/home/zzd/project/robot1_zzd_test_bundle/rtc_python_optimized_20260917.tar.gz` 更新。

'''
if notice not in readme.read_text():
    readme.write_text(notice + readme.read_text())
shutil.copy2(src / 'README_RTC_ENGINEERING_OPTIMIZATIONS.md', package / 'README_RTC_ENGINEERING_OPTIMIZATIONS.md')
print('Updated 113 legacy package builder and README, with backups.')
