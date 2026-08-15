$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$Root = 'D:\Reddit-Romantics\Automation\videos'
New-Item -ItemType Directory -Force -Path $Root | Out-Null
$Log = Join-Path $Root 'download.log'
$Csv = Join-Path $Root 'licenses.csv'
$Sources = Join-Path $Root 'SOURCES.md'

$items = @(
  @{Title='Pottery_02.webm'; Category='pottery'; Priority='verified'},
  @{Title='Pinching_Method_in_Ceramics.webm'; Category='clay'; Priority='verified'},
  @{Title='Pottery_on_porters_wheel.webm'; Category='pottery'; Priority='verified'},
  @{Title='Making_of_Terracotta_in_Asharikandi_Village,_Dhubri,_Assam.webm'; Category='clay'; Priority='verified'},
  @{Title='Clay_and_the_community.webm'; Category='clay'; Priority='verified'},
  @{Title='Nigeria_Liquid_Soap_Making.webm'; Category='soap'; Priority='verified'},
  @{Title='MASTER_CARVER_AT_WORK_By_Green_Wilfred_Somoni.webm'; Category='carving'; Priority='verified'},
  @{Title='From_Old_To_New_(The_Shoemaking_Process).webm'; Category='restoration'; Priority='verified'},
  @{Title='A_Blacksmith_at_work.webm'; Category='manufacturing'; Priority='verified'},
  @{Title='Glass_blowing-_forming_the_body,_part_1.webm'; Category='manufacturing'; Priority='verified'},
  @{Title='(ASMR)_The_Scratching_Pencil,_Episode_1-_Spirograph.webm'; Category='drawing'; Priority='verified'},
  @{Title='Embroidery.webm'; Category='miscellaneous'; Priority='verified'},
  @{Title='Can_crusher.webm'; Category='miscellaneous'; Priority='verified'},
  @{Title='Sculpture_Molding.webm'; Category='clay'; Priority='verified'},

  @{Title='Pottery_1.webm'; Category='pottery'; Priority='candidate'},
  @{Title='Palm_oil_production_documentation.webm'; Category='manufacturing'; Priority='candidate'},
  @{Title='Palmoil_Production_-_Wikiloves_Africa_2024.webm'; Category='manufacturing'; Priority='candidate'},
  @{Title='Full_wood_curving_process-.webm'; Category='carving'; Priority='candidate'},
  @{Title='The_Master_Carver_and_a_Message_-_A_Documentary_By_Green_Wilfred_Somoni.webm'; Category='carving'; Priority='candidate'},
  @{Title='The_Process.webm'; Category='manufacturing'; Priority='candidate'},
  @{Title='Slippers_making_01.webm'; Category='manufacturing'; Priority='candidate'},
  @{Title='Slippers_making_26.webm'; Category='manufacturing'; Priority='candidate'},
  @{Title='Aluminium_pot-making_in_a_foundry_in_Kaduna.webm'; Category='manufacturing'; Priority='candidate'},
  @{Title='Beads_maker_2.webm'; Category='miscellaneous'; Priority='candidate'},
  @{Title='Canvas_Painting.webm'; Category='drawing'; Priority='candidate'},
  @{Title='Plasterboard_design_001.webm'; Category='manufacturing'; Priority='candidate'},
  @{Title='Plasterboard_making.webm'; Category='manufacturing'; Priority='candidate'},
  @{Title='Wood_Chopping.webm'; Category='miscellaneous'; Priority='candidate'},
  @{Title='Tyre_Fix_Full_Video.webm'; Category='restoration'; Priority='candidate'},
  @{Title='Tyre_Fix_Short_Video.webm'; Category='restoration'; Priority='candidate'}
)

$allowLicenses = @('CC BY 3.0','CC BY 4.0','CC BY-SA 3.0','CC BY-SA 4.0','CC0','Public domain','Public Domain')
$records = New-Object System.Collections.Generic.List[object]

function Strip-Html([string]$s) {
  if ([string]::IsNullOrWhiteSpace($s)) { return '' }
  return [System.Net.WebUtility]::HtmlDecode(($s -replace '<[^>]+>','' -replace '\s+',' ')).Trim()
}

function Log([string]$msg) {
  $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $msg"
  Add-Content -LiteralPath $Log -Value $line -Encoding UTF8
  Write-Host $line
}

foreach ($it in $items) {
  $title = $it.Title
  $apiTitle = 'File:' + $title
  $api = 'https://commons.wikimedia.org/w/api.php?action=query&format=json&prop=imageinfo&iiprop=url%7Csize%7Cmime%7Cextmetadata&titles=' + [uri]::EscapeDataString($apiTitle)
  try {
    $resp = Invoke-RestMethod -Uri $api -Headers @{'User-Agent'='ChatGPT-Codex-SatisfyingVideoDownloader/1.0'}
    $page = $resp.query.pages.PSObject.Properties.Value | Select-Object -First 1
    if (-not $page.imageinfo) { Log "SKIP no-imageinfo $title"; continue }
    $ii = $page.imageinfo[0]
    $m = $ii.extmetadata
    $license = if ($m.LicenseShortName) { [string]$m.LicenseShortName.value } else { '' }
    $licenseUrl = if ($m.LicenseUrl) { [string]$m.LicenseUrl.value } else { '' }
    $restrictions = if ($m.Restrictions) { [string]$m.Restrictions.value } else { '' }
    $artist = if ($m.Artist) { Strip-Html ([string]$m.Artist.value) } else { '' }
    $credit = if ($m.Credit) { Strip-Html ([string]$m.Credit.value) } else { '' }
    $desc = if ($m.ImageDescription) { Strip-Html ([string]$m.ImageDescription.value) } else { '' }
    $duration = [double]$ii.duration
    $okLicense = $allowLicenses -contains $license
    $ok = ($duration -ge 30) -and $okLicense -and [string]::IsNullOrWhiteSpace($restrictions) -and ($ii.mime -like 'video/*')

    if (-not $ok) {
      Log "SKIP gate-failed $title duration=$duration license='$license' restrictions='$restrictions' mime='$($ii.mime)'"
      continue
    }

    $dir = Join-Path $Root $it.Category
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    $filename = [uri]::UnescapeDataString(($ii.url -split '/')[-1].Split('?')[0])
    if ([string]::IsNullOrWhiteSpace($filename)) { $filename = $title }
    $dest = Join-Path $dir $filename
    $expected = [int64]$ii.size

    $needDownload = $true
    if (Test-Path -LiteralPath $dest) {
      $actual = (Get-Item -LiteralPath $dest).Length
      if ($actual -eq $expected) {
        $needDownload = $false
        Log "OK existing $filename ($actual bytes)"
      } else {
        Log "REDOWNLOAD size mismatch $filename actual=$actual expected=$expected"
      }
    }

    if ($needDownload) {
      Log "DOWNLOAD $filename duration=$([math]::Round($duration,1))s size=$expected license='$license'"
      $tmp = $dest + '.part'
      & curl.exe -L --fail --retry 8 --retry-delay 10 --connect-timeout 20 -C - -A 'ChatGPT-Codex-SatisfyingVideoDownloader/1.0' -o $tmp $ii.url
      if ($LASTEXITCODE -ne 0) { throw "curl failed with exit code $LASTEXITCODE" }
      $got = (Get-Item -LiteralPath $tmp).Length
      if ($got -ne $expected) { throw "Downloaded size mismatch for ${filename}: got=$got expected=$expected" }
      Move-Item -LiteralPath $tmp -Destination $dest -Force
      Log "DONE $filename"
      Start-Sleep -Seconds 3
    }

    $records.Add([pscustomobject]@{
      filename=$filename
      category=$it.Category
      priority=$it.Priority
      source_url=$ii.descriptionurl
      original_download_url=$ii.url
      creator=$artist
      credit=$credit
      license=$license
      license_url=$licenseUrl
      duration_seconds=[math]::Round($duration,3)
      width=$ii.width
      height=$ii.height
      bytes=$expected
      download_date=(Get-Date -Format 'yyyy-MM-dd')
      notes=$desc
    })
  }
  catch {
    Log "ERROR $title :: $($_.Exception.Message)"
  }
}

$records | Sort-Object category,filename | Export-Csv -LiteralPath $Csv -NoTypeInformation -Encoding UTF8

$md = New-Object System.Collections.Generic.List[string]
$md.Add('# Satisfying/process video sources')
$md.Add('')
$md.Add('Downloaded from Wikimedia Commons as original media files after checking Commons API metadata for duration, MIME type, reuse license, and restrictions.')
$md.Add('')
$md.Add("Generated: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')")
$md.Add('')
foreach ($r in ($records | Sort-Object category,filename)) {
  $md.Add("## $($r.filename)")
  $md.Add("")
  $md.Add("- Category: $($r.category)")
  $md.Add("- Duration: $($r.duration_seconds) seconds")
  $md.Add("- Resolution: $($r.width)x$($r.height)")
  $md.Add("- Creator: $($r.creator)")
  $md.Add("- License: $($r.license)")
  $md.Add("- License URL: $($r.license_url)")
  $md.Add("- Source: $($r.source_url)")
  $md.Add("- Original file: $($r.original_download_url)")
  if ($r.notes) { $md.Add("- Notes: $($r.notes)") }
  $md.Add('')
}
$md | Set-Content -LiteralPath $Sources -Encoding UTF8
Log "FINISHED files=$($records.Count) csv='$Csv' sources='$Sources'"
