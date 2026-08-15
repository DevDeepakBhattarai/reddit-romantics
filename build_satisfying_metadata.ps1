$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
$Root='D:\Reddit-Romantics\Automation\videos'
$items=@(
 @{Title='Pottery_02.webm';Category='pottery'},
 @{Title='Pinching_Method_in_Ceramics.webm';Category='clay'},
 @{Title='Pottery_on_porters_wheel.webm';Category='pottery'},
 @{Title='Making_of_Terracotta_in_Asharikandi_Village,_Dhubri,_Assam.webm';Category='clay'},
 @{Title='Clay_and_the_community.webm';Category='clay'},
 @{Title='Nigeria_Liquid_Soap_Making.webm';Category='soap'},
 @{Title='MASTER_CARVER_AT_WORK_By_Green_Wilfred_Somoni.webm';Category='carving'},
 @{Title='From_Old_To_New_(The_Shoemaking_Process).webm';Category='restoration'},
 @{Title='A_Blacksmith_at_work.webm';Category='manufacturing'},
 @{Title='Glass_blowing-_forming_the_body,_part_1.webm';Category='manufacturing'},
 @{Title='(ASMR)_The_Scratching_Pencil,_Episode_1-_Spirograph.webm';Category='drawing'},
 @{Title='Embroidery.webm';Category='miscellaneous'},
 @{Title='Can_crusher.webm';Category='miscellaneous'},
 @{Title='Sculpture_Molding.webm';Category='clay'}
)
function Strip-Html([string]$s){if(!$s){return ''}; return [System.Net.WebUtility]::HtmlDecode(($s -replace '<[^>]+>','' -replace '\s+',' ')).Trim()}
$rows=@()
foreach($it in $items){
 $path=Join-Path (Join-Path $Root $it.Category) $it.Title
 if(!(Test-Path -LiteralPath $path)){continue}
 $api='https://commons.wikimedia.org/w/api.php?action=query&format=json&prop=imageinfo&iiprop=url%7Csize%7Cmime%7Cextmetadata&titles='+[uri]::EscapeDataString('File:'+$it.Title)
 $r=Invoke-RestMethod -Uri $api -Headers @{'User-Agent'='ChatGPT-Codex-SatisfyingVideoDownloader/1.0'}
 $ii=($r.query.pages.PSObject.Properties.Value|Select-Object -First 1).imageinfo[0]
 $m=$ii.extmetadata
 $rows += [pscustomobject]@{
  filename=$it.Title;category=$it.Category;source_url=$ii.descriptionurl;original_download_url=$ii.url;
  creator=if($m.Artist){Strip-Html([string]$m.Artist.value)}else{''};
  license=if($m.LicenseShortName){[string]$m.LicenseShortName.value}else{''};
  license_url=if($m.LicenseUrl){[string]$m.LicenseUrl.value}else{''};
  duration_seconds=[math]::Round([double]$ii.duration,3);width=$ii.width;height=$ii.height;bytes=(Get-Item -LiteralPath $path).Length;
  sha256=(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash;download_date=(Get-Date -Format 'yyyy-MM-dd');
  notes=if($m.ImageDescription){Strip-Html([string]$m.ImageDescription.value)}else{''}
 }
}
$rows|Export-Csv -LiteralPath (Join-Path $Root 'licenses.csv') -NoTypeInformation -Encoding UTF8
$md=@('# Satisfying/process video sources','','These files were downloaded as the original Wikimedia Commons media assets. License/source data below comes from Commons API metadata.','')
foreach($x in $rows){
 $md += "## $($x.filename)",'',"- Category: $($x.category)","- Duration: $($x.duration_seconds) seconds","- Resolution: $($x.width)x$($x.height)","- Creator: $($x.creator)","- License: $($x.license)","- License URL: $($x.license_url)","- Source: $($x.source_url)","- Original file: $($x.original_download_url)","- SHA-256: $($x.sha256)","- Notes: $($x.notes)",''
}
$md|Set-Content -LiteralPath (Join-Path $Root 'SOURCES.md') -Encoding UTF8
$rows|Select-Object filename,duration_seconds,width,height,license,bytes|Format-Table -AutoSize
