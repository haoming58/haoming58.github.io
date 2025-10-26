# 创建目标目录
$sourceDir = "_notes\deep-learning\rnn\figures"
$destDir = "assets\img\notes\rnn\figures"

# 创建目标目录（如果不存在）
if (-not (Test-Path $destDir)) {
    New-Item -Path $destDir -ItemType Directory -Force
    Write-Host "创建目录: $destDir"
}

# 复制所有图片文件
if (Test-Path $sourceDir) {
    Copy-Item -Path "$sourceDir\*" -Destination $destDir -Recurse -Force
    Write-Host "图片已复制从 $sourceDir 到 $destDir"
    Write-Host "复制完成！"
} else {
    Write-Host "源目录不存在: $sourceDir"
}
