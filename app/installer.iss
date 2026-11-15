; 터링 자동 채점기 설치 프로그램 (Inno Setup 6)
; 먼저 app\build_exe.py 로 dist\터링채점기 폴더를 만든 뒤 컴파일한다:
;   "%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe" app\installer.iss
; 결과: dist\터링채점기_설치.exe (이 파일 하나만 전달하면 된다)

#define AppName "터링 자동 채점기"
#define AppExe "터링채점기.exe"
#define AppVersion "1.0.0"

[Setup]
AppId={{6B0E5A52-3C0D-4E0B-9A57-7E2D1C9F4A11}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
; 관리자 권한 없이 현재 사용자 계정에만 설치
PrivilegesRequired=lowest
DefaultDirName={localappdata}\Programs\터링채점기
DisableProgramGroupPage=yes
DisableDirPage=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..\dist
OutputBaseFilename=터링채점기_설치
Compression=lzma2/fast
SolidCompression=no
WizardStyle=modern
UninstallDisplayName={#AppName}
UninstallDisplayIcon={app}\{#AppExe}

[Languages]
Name: "korean"; MessagesFile: "compiler:Languages\Korean.isl"

[Tasks]
Name: "desktopicon"; Description: "바탕화면에 바로가기 만들기"; GroupDescription: "추가 작업:"

[Files]
Source: "..\dist\터링채점기\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\{#AppExe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExe}"; Description: "{#AppName} 실행"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: files; Name: "{app}\error_log.txt"
