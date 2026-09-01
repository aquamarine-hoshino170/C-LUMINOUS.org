const fs = require('fs');
const path = require('path');
const AdmZip = require('adm-zip');

const zip = new AdmZip();

// [Content_Types].xml তৈরি
const contentTypes = `<?xml version="1.0" encoding="utf-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="json" ContentType="application/json" />
  <Default Extension="vsixmanifest" ContentType="text/xml" />
</Types>`;
zip.addFile('[Content_Types].xml', Buffer.from(contentTypes, 'utf8'));

// extension.vsixmanifest তৈরি
const manifest = `<?xml version="1.0" encoding="utf-8"?>
<PackageManifest Version="2.0.0" xmlns="http://schemas.microsoft.com/developer/vsx-schema/2011">
  <Metadata>
    <Identity Id="luminous-lang" Version="1.0.0" Publisher="luminous-team" />
    <DisplayName>Luminous Language Support</DisplayName>
    <Description>Official syntax highlighting and snippets for Luminous</Description>
  </Metadata>
  <Installation>
    <InstallationTarget Id="Microsoft.VisualStudio.Code"/>
  </Installation>
  <Dependencies/>
  <Assets>
    <Asset Type="Microsoft.VisualStudio.Code.Manifest" Path="extension/package.json" Addressable="true" />
  </Assets>
</PackageManifest>`;
zip.addFile('extension.vsixmanifest', Buffer.from(manifest, 'utf8'));

// এক্সটেনশনের ফাইলগুলো extension/ সাব-ডিরেক্টরিতে যোগ করা
zip.addLocalFile('package.json', 'extension');
zip.addLocalFile('language-configuration.json', 'extension');
zip.addLocalFolder('syntaxes', 'extension/syntaxes');
zip.addLocalFolder('snippets', 'extension/snippets');

// VSIX সেভ করা
const outputFile = 'luminous-lang-1.0.0.vsix';
zip.writeZip(outputFile);
console.log(`\x1b[1;32m🎉 Successfully built: ${outputFile}\x1b[0m`);
