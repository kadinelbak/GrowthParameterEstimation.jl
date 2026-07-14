 = Get-Content src/data.jl
 = False
 = -1
 = -1
for ( = 0;  -lt .Length; ++) {
    if ([].Trim() -eq '"""') {
        if (-not ) {
             = True
             = 
        } else {
             = False
             = 
        }
    }
}
if ( -eq -1 -or  -eq -1) {
    Write-Error \"Docstring not found\"
    exit 1
}
 = @()
 = False
for ( = 0;  -lt .Length; ++) {
    if ( -ge  -and  -le ) {
        if ([].Trim() -eq 'normalize_schema(df::DataFrame; column_map=nothing, defaults=nothing)') {
             = True
            continue
        }
         += []
    } else {
         += []
    }
}
if (-not ) {
    Write-Error \"Erroneous line not found in docstring\"
    exit 1
}
 = -1
for ( = ;  -lt .Length; ++) {
    if ([].Trim() -eq '"""') {
         = 
        break
    }
}
if ( -eq -1) {
    Write-Error \"Could not find closing triple quotes in new lines\"
    exit 1
}
 = @()
for ( = 0;  -le ; ++) {
     += []
}
 += 'function normalize_schema(df::DataFrame; column_map=nothing, defaults=nothing)'
for ( = +1;  -lt .Length; ++) {
     += []
}
Set-Content src/data.jl 
