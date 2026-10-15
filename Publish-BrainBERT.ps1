#requires -Version 5.1
[CmdletBinding()]
param(
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$')]
    [string]$RepoName = 'brainbert-lab',
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9-]{0,38}$')]
    [string]$ExpectedOwner = 'REcklessNOLOve117'
)

# Run this script from the extracted publication package. It creates a PUBLIC
# GitHub repository and publishes docs/ with GitHub Pages. It never force-pushes.
# A same-name repository is accepted only when this package's private local
# state records this script's creation and origin still points to that repository.
Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'
$Root = [System.IO.Path]::GetFullPath($PSScriptRoot)
$StatePath = Join-Path $Root '.brainbert-publish-state.json'
$ScriptLeaf = Split-Path -Leaf $PSCommandPath
$FullName = $ExpectedOwner + '/' + $RepoName
$RepoUrl = 'https://github.com/' + $FullName
$Utf8NoBom = New-Object System.Text.UTF8Encoding($false)
$State = $null
$PayloadPath = $null
$PushedLocation = $false
$SavedEnvironment = @{}

function Invoke-Native {
    param([string]$Executable, [string[]]$Arguments)
    $previous = $ErrorActionPreference
    try {
        # Windows PowerShell 5.1 wraps redirected native stderr as ErrorRecord.
        # Capture it without turning a harmless native stderr line into a throw.
        $ErrorActionPreference = 'Continue'
        $PSNativeCommandUseErrorActionPreference = $false
        $captured = @(& $Executable @Arguments 2>&1 | ForEach-Object { $_.ToString() })
        $code = $LASTEXITCODE
    }
    finally { $ErrorActionPreference = $previous }
    return [pscustomobject]@{ Code = $code; Text = ($captured -join [Environment]::NewLine) }
}

function Assert-Native {
    param($Result, [string]$Operation)
    if ($Result.Code -ne 0) {
        $detail = $Result.Text
        if ($detail.Length -gt 2500) { $detail = $detail.Substring(0, 2500) }
        throw ($Operation + ' failed (exit ' + $Result.Code + ').' + [Environment]::NewLine + $detail)
    }
}

function Git-Checked {
    param([string[]]$Arguments)
    $result = Invoke-Native $script:GitExe $Arguments
    Assert-Native $result ('git ' + ($Arguments -join ' '))
    return $result.Text
}

function Gh-Api {
    param([string]$Endpoint, [string]$Method = 'GET', [string]$InputFile = '')
    $arguments = @('api', '--hostname', 'github.com', '--method', $Method,
        '-H', 'Accept: application/vnd.github+json',
        '-H', 'X-GitHub-Api-Version: 2022-11-28', $Endpoint)
    if ($InputFile) { $arguments += @('--input', $InputFile) }
    return (Invoke-Native $script:GhExe $arguments)
}

function Has-HttpStatus {
    param($Result, [int]$Status)
    return ($Result.Code -ne 0 -and $Result.Text -match ('\bHTTP\s+' + $Status + '\b'))
}

function Json-FromResult {
    param($Result, [string]$Operation)
    Assert-Native $Result $Operation
    try { return ($Result.Text | ConvertFrom-Json) }
    catch { throw ($Operation + ' did not return valid JSON.') }
}

function Get-Repository {
    $result = Gh-Api ('repos/' + $script:FullName)
    if (Has-HttpStatus $result 404) { return $null }
    return (Json-FromResult $result 'Read GitHub repository')
}

function Property-Value {
    param($Object, [string]$Name)
    if ($null -eq $Object) { return $null }
    $property = $Object.PSObject.Properties[$Name]
    if ($null -eq $property) { return $null }
    return $property.Value
}

function Origin-Matches {
    param([string]$Url)
    $normalized = $Url.Trim().TrimEnd('/')
    $normalized = $normalized -replace '^git@github\.com:', 'https://github.com/'
    $normalized = $normalized -replace '^ssh://git@github\.com/', 'https://github.com/'
    $normalized = $normalized -replace '\.git$', ''
    return ($normalized -ieq $script:RepoUrl)
}

function Assert-Origin {
    $fetch = Invoke-Native $script:GitExe @('remote', 'get-url', 'origin')
    Assert-Native $fetch 'Read origin URL'
    $push = Invoke-Native $script:GitExe @('remote', 'get-url', '--push', 'origin')
    Assert-Native $push 'Read origin push URL'
    if (-not (Origin-Matches $fetch.Text) -or -not (Origin-Matches $push.Text)) {
        throw 'Origin does not match the requested GitHub repository. Nothing was pushed.'
    }
}

function Save-State {
    $json = $script:State | ConvertTo-Json -Depth 6
    [System.IO.File]::WriteAllText($script:StatePath, $json, $script:Utf8NoBom)
}

function Assert-Repository {
    param($Repository)
    if ($Repository.full_name -ine $script:FullName -or $Repository.owner.login -ine $script:ExpectedOwner) {
        throw 'GitHub returned an unexpected repository owner or name.'
    }
    if ($Repository.private) { throw 'The existing repository is private; this script will not change its visibility.' }
    $recordedId = Property-Value $script:State 'RepositoryId'
    if ($null -ne $recordedId -and [string]$recordedId -ne [string]$Repository.id) {
        throw 'The repository ID changed since this package created it. Refusing to continue.'
    }
}

function Set-LocalAuthor {
    param([string]$Field, [string]$Fallback)
    $result = Invoke-Native $script:GitExe @('config', '--local', '--get', $Field)
    if ($result.Code -eq 1 -or ($result.Code -eq 0 -and -not $result.Text.Trim())) {
        $null = Git-Checked @('config', '--local', $Field, $Fallback)
    }
    else { Assert-Native $result ('Read repository-local ' + $Field) }
}

function Assert-PagesSource {
    param($Pages)
    $source = Property-Value $Pages 'source'
    if ((Property-Value $source 'branch') -ne 'main' -or (Property-Value $source 'path') -ne '/docs') {
        throw 'Pages already uses another publishing source. Its settings were preserved; expected main /docs.'
    }
    $buildType = Property-Value $Pages 'build_type'
    if ($buildType -and $buildType -ne 'legacy') {
        throw 'Pages uses a custom workflow. Its settings were preserved; this package expects branch publishing.'
    }
}

try {
    foreach ($name in @('GH_HOST', 'GH_PROMPT_DISABLED', 'GIT_TERMINAL_PROMPT')) {
        $SavedEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
    }
    $env:GH_HOST = 'github.com'
    $env:GH_PROMPT_DISABLED = '1'
    $env:GIT_TERMINAL_PROMPT = '0'
    Push-Location $Root
    $PushedLocation = $true

    $ghCommand = Get-Command gh -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    $gitCommand = Get-Command git -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($null -eq $ghCommand -or $null -eq $gitCommand) {
        throw 'Install Git for Windows and GitHub CLI (gh), then reopen PowerShell and run this script again.'
    }
    $script:GhExe = $ghCommand.Source
    $script:GitExe = $gitCommand.Source
    Assert-Native (Invoke-Native $GitExe @('--version')) 'Check git'
    Assert-Native (Invoke-Native $GhExe @('--version')) 'Check gh'
    $auth = Invoke-Native $GhExe @('auth', 'status', '--hostname', 'github.com')
    if ($auth.Code -ne 0) { throw 'GitHub CLI is not authenticated. Run: gh auth login --hostname github.com' }
    # Do not print auth status or tokens. The API identifies the active account.
    $user = Json-FromResult (Gh-Api 'user') 'Read authenticated GitHub account'
    if ($user.login -ine $ExpectedOwner) {
        throw ('Active GitHub account is ' + $user.login + '; expected ' + $ExpectedOwner + '. Run gh auth switch --hostname github.com --user ' + $ExpectedOwner)
    }

    foreach ($required in @('docs/index.html', 'README.md', 'THIRD_PARTY_NOTICES.md', 'scripts/prepare_data.py', $ScriptLeaf)) {
        if (-not (Test-Path -LiteralPath (Join-Path $Root $required) -PathType Leaf)) {
            throw ('Missing package file: ' + $required + '. Extract the complete ZIP before running this script.')
        }
    }
    Write-Host ('Account: ' + $user.login)
    Write-Host ('Public repository: ' + $RepoUrl)
    Write-Host 'The docs/ website and publication files will be publicly readable.'

    $hasGit = Test-Path -LiteralPath (Join-Path $Root '.git')
    if (Test-Path -LiteralPath $StatePath -PathType Leaf) {
        $State = Get-Content -LiteralPath $StatePath -Raw -Encoding UTF8 | ConvertFrom-Json
        if ($State.Format -ne 'BrainBERTLocalPublisherV1' -or $State.Owner -ine $ExpectedOwner -or
            $State.Repo -ine $RepoName -or $State.Root -ine $Root) {
            throw 'Local publication state does not match this folder and requested repository.'
        }
    }
    elseif ($hasGit) {
        throw 'This folder already contains a Git repository without this publisher state. Use a fresh extracted package.'
    }

    $repository = Get-Repository
    if ($null -ne $repository) {
        if ($null -eq $State -or -not $hasGit) {
            throw 'A same-name GitHub repository already exists. It was not created by this package. Choose a different -RepoName.'
        }
        Assert-Origin
        # gh adds origin only AFTER creating a new repository. A recorded
        # in-progress creation plus this matching origin recovers interruption
        # between remote creation and state save, including an initial push error.
        if (-not $State.CreatedByScript -and $State.Phase -ne 'creating') {
            throw 'The existing repository is not recorded as created by this publisher. Refusing to reuse it.'
        }
        Assert-Repository $repository
        $State.CreatedByScript = $true
        $State.RepositoryId = $repository.id
        $State.Phase = 'created'
        Save-State
        Write-Host 'Resuming this package and its matching repository.'
    }
    else {
        if ($null -ne $State -and $State.CreatedByScript) {
            throw 'The repository previously created by this package is now missing or inaccessible. Refusing to recreate it silently.'
        }
        if ($null -eq $State) {
            $State = [pscustomobject]@{
                Format = 'BrainBERTLocalPublisherV1'; Owner = $ExpectedOwner; Repo = $RepoName
                Root = $Root; CreatedByScript = $false; RepositoryId = $null
                Phase = 'prepared'; StartedAt = [DateTime]::UtcNow.ToString('o')
            }
            Save-State
        }
        if (-not $hasGit) {
            $null = Git-Checked @('init')
            $null = Git-Checked @('symbolic-ref', 'HEAD', 'refs/heads/main')
        }
        $remotes = Git-Checked @('remote')
        if ($remotes.Trim()) { throw 'A pre-creation local repository already has remotes. Refusing to replace them.' }
    }

    $branch = Git-Checked @('symbolic-ref', '--short', 'HEAD')
    if ($branch.Trim() -ne 'main') { throw 'The current branch is not main. No branch was switched automatically.' }
    Set-LocalAuthor 'user.name' $ExpectedOwner
    $fallbackEmail = '74002384+REcklessNOLOve117@users.noreply.github.com'
    if ($ExpectedOwner -ine 'REcklessNOLOve117') {
        $fallbackEmail = [string]$user.id + '+' + $user.login + '@users.noreply.github.com'
    }
    Set-LocalAuthor 'user.email' $fallbackEmail

    # Reset inherited credential helpers for github.com, then use the active
    # gh login. This changes this repository's config only and stores no token.
    # Write the empty helper as Git config text: PowerShell 5.1 drops empty
    # native arguments, so `git config ... ''` is not portable across PS versions.
    $gitConfigPath = (Git-Checked @('rev-parse', '--git-path', 'config')).Trim()
    $credentialMarker = '# BrainBERT publisher: repository-local GitHub CLI credentials'
    $gitConfigText = [System.IO.File]::ReadAllText([System.IO.Path]::GetFullPath($gitConfigPath))
    if (-not $gitConfigText.Contains($credentialMarker)) {
        $credentialBlock = "`r`n" + $credentialMarker + "`r`n" +
            '[credential "https://github.com"]' + "`r`n" +
            "`thelper =`r`n`thelper = !gh auth git-credential`r`n"
        [System.IO.File]::AppendAllText([System.IO.Path]::GetFullPath($gitConfigPath), $credentialBlock, $Utf8NoBom)
    }
    $credentialCheck = Invoke-Native $GitExe @('config', '--local', '--get-all', 'credential.https://github.com.helper')
    Assert-Native $credentialCheck 'Check repository-local GitHub credential helper'

    $ignorePath = Join-Path $Root '.gitignore'
    $ignoreText = ''
    if (Test-Path -LiteralPath $ignorePath -PathType Leaf) { $ignoreText = [System.IO.File]::ReadAllText($ignorePath) }
    if ($ignoreText -notmatch '(?m)^/?\.brainbert-publish-state\.json\s*$') {
        $ignoreText = $ignoreText.TrimEnd() + "`r`n/.brainbert-publish-state.json`r`n"
        [System.IO.File]::WriteAllText($ignorePath, $ignoreText, $Utf8NoBom)
    }
    $noJekyll = Join-Path $Root 'docs/.nojekyll'
    if (-not (Test-Path -LiteralPath $noJekyll)) { [System.IO.File]::WriteAllText($noJekyll, '', $Utf8NoBom) }
    # Explicit publication paths; no blanket git add .
    $null = Git-Checked @('add', '--', 'docs', 'scripts', 'README.md', 'THIRD_PARTY_NOTICES.md', $ScriptLeaf, '.gitignore')
    $staged = Git-Checked @('diff', '--cached', '--name-only')
    foreach ($path in ($staged -split '\r?\n')) {
        if ($path -and $path -notmatch '^(docs/|scripts/)' -and
            $path -notin @('README.md', 'THIRD_PARTY_NOTICES.md', $ScriptLeaf, '.gitignore')) {
            throw ('An unrelated file is already staged: ' + $path + '. Unstage it before continuing.')
        }
    }
    $changes = Invoke-Native $GitExe @('diff', '--cached', '--quiet')
    if ($changes.Code -eq 1) { $null = Git-Checked @('commit', '-m', 'Publish BrainBERT interactive learning lab') }
    elseif ($changes.Code -ne 0) { Assert-Native $changes 'Check staged changes' }
    $commit = (Git-Checked @('rev-parse', 'HEAD')).Trim()

    if ($null -eq $repository) {
        $State.Phase = 'creating'
        Save-State
        $creation = Invoke-Native $GhExe @('repo', 'create', $FullName, '--public', '--source', '.', '--remote', 'origin', '--push',
            '--description', 'Interactive BrainBERT learning lab: STFT, tokens, attention, masking and fine-tuning')
        $origin = Invoke-Native $GitExe @('remote', 'get-url', 'origin')
        if ($creation.Code -eq 0 -or ($origin.Code -eq 0 -and (Origin-Matches $origin.Text))) {
            # Save proof before any subsequent network request can fail.
            $State.CreatedByScript = $true
            $State.Phase = 'created'
            Save-State
        }
        Assert-Native $creation 'Create public GitHub repository and push (rerun this script after resolving any push/authentication error)'
        Assert-Origin
        $repository = Get-Repository
        if ($null -eq $repository) { throw 'Repository creation returned success, but the repository is not yet readable. Rerun this script.' }
        Assert-Repository $repository
        $State.RepositoryId = $repository.id
        Save-State
    }
    Assert-Origin
    # No force, no replacement of an existing unrelated history.
    $null = Git-Checked @('push', 'origin', 'HEAD:refs/heads/main')
    Write-Host ('Source pushed: ' + $commit.Substring(0, 12))

    $pagesEndpoint = 'repos/' + $FullName + '/pages'
    $pagesResult = Gh-Api $pagesEndpoint
    if (Has-HttpStatus $pagesResult 404) {
        $PayloadPath = [System.IO.Path]::GetTempFileName()
        $payload = @{ build_type = 'legacy'; source = @{ branch = 'main'; path = '/docs' } } | ConvertTo-Json -Depth 4
        [System.IO.File]::WriteAllText($PayloadPath, $payload, $Utf8NoBom)
        try { $createdPages = Gh-Api $pagesEndpoint 'POST' $PayloadPath }
        finally {
            if (Test-Path -LiteralPath $PayloadPath) { Remove-Item -LiteralPath $PayloadPath -Force }
            $PayloadPath = $null
        }
        if ($createdPages.Code -eq 0) { $pages = Json-FromResult $createdPages 'Enable GitHub Pages' }
        elseif ((Has-HttpStatus $createdPages 409) -or (Has-HttpStatus $createdPages 422)) {
            # A parallel request may already have enabled Pages; read and verify.
            $pages = Json-FromResult (Gh-Api $pagesEndpoint) 'Read existing GitHub Pages settings'
        }
        else { Assert-Native $createdPages 'Enable GitHub Pages' }
    }
    else { $pages = Json-FromResult $pagesResult 'Read GitHub Pages settings' }
    Assert-PagesSource $pages
    $State.Phase = 'pages-configured'
    Save-State

    Write-Host 'Waiting for the build of this commit (up to 60 seconds)...'
    $watch = [System.Diagnostics.Stopwatch]::StartNew()
    $built = $false
    $lastStatus = 'not started'
    while ($watch.Elapsed.TotalSeconds -lt 60) {
        $buildResult = Gh-Api ($pagesEndpoint + '/builds/latest')
        if ($buildResult.Code -eq 0) {
            $build = Json-FromResult $buildResult 'Read latest Pages build'
            $lastStatus = [string]$build.status
            # Never mistake an older successful build for the new commit.
            if ($build.commit -eq $commit -and $build.status -eq 'built') { $built = $true; break }
            if ($build.commit -eq $commit -and $build.status -eq 'errored') {
                $buildError = Property-Value (Property-Value $build 'error') 'message'
                throw ('GitHub Pages build failed: ' + [string]$buildError + '. Check ' + $RepoUrl + '/actions')
            }
        }
        elseif (-not (Has-HttpStatus $buildResult 404) -and -not (Has-HttpStatus $buildResult 409)) {
            Assert-Native $buildResult 'Read latest Pages build'
        }
        $remaining = 60 - $watch.Elapsed.TotalSeconds
        if ($remaining -gt 0) { Start-Sleep -Milliseconds ([int][Math]::Min(5000, [Math]::Ceiling($remaining * 1000))) }
    }
    if ($built) {
        $pages = Json-FromResult (Gh-Api $pagesEndpoint) 'Read published Pages URL'
        Assert-PagesSource $pages
        if (-not $pages.html_url) { throw 'The build succeeded but GitHub did not return a Pages URL.' }
        $State.Phase = 'published'
        Save-State
        Write-Host ''
        Write-Host 'Published successfully:' -ForegroundColor Green
        Write-Host $pages.html_url
    }
    else {
        Write-Host ''
        Write-Host ('Source submitted; Pages is not confirmed ready yet (latest status: ' + $lastStatus + ').') -ForegroundColor Yellow
        Write-Host ('Check shortly: ' + $RepoUrl + '/settings/pages')
        if ($pages.html_url) { Write-Host ('Expected URL (not yet confirmed ready): ' + $pages.html_url) }
    }
}
catch {
    Write-Host ''
    Write-Host ('Publication did not finish: ' + $_.Exception.Message) -ForegroundColor Red
    Write-Host 'Your GitHub token was not printed. The script does not delete repositories or force-push.'
    exit 1
}
finally {
    if ($PayloadPath -and (Test-Path -LiteralPath $PayloadPath)) { Remove-Item -LiteralPath $PayloadPath -Force -ErrorAction SilentlyContinue }
    if ($PushedLocation) { Pop-Location }
    foreach ($name in $SavedEnvironment.Keys) {
        [Environment]::SetEnvironmentVariable($name, $SavedEnvironment[$name], 'Process')
    }
}
