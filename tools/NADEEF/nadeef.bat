:begin
@echo off

:START
set argC=0
for %%x in (%*) do Set /A argC+=1

if not %argC% == 1 (
    "java" -cp out\bin\*;out\test;examples\;out\production\ qa.qcri.nadeef.console.Console
) else (
    if "%1" == "console" (
        "java" -cp out\bin\*;out\test;examples\;out\production\ qa.qcri.nadeef.console.Console
    ) else if "%1" == "dashboard" (
        "java" -cp out\bin\*;out\production\ qa.qcri.nadeef.web.NadeefStart
    ) else (
        goto usage
    )
)

goto end

:noJDK
echo Cannot find JDK installed.
goto end

:noCompile
echo Nadeef is not compiled yet. Run 'ant' to compile Nadeef.

:usage
echo Usage: nadeef.sh [OPTIONS]
echo Options are:
echo     console : start the NADEEF console.
echo     dashboard : start the NADEEF dashboard.

:end
