-- BG -> index 0 with STABLE ROTATION of palette order; pad/crop to NxN; save to out/
-- Modes: Auto | Pick from sample | Pick manual | Alpha
-- Headless (-b) params:
--   in=PATH [recursive=1] [size=256] [mode=auto|sample|manual|alpha] [bg=#ff00ff] [trans0=0|1]

local fs = app.fs

-- ---------- helpers ----------
local function hexToColor(hex)
  local r,g,b = tostring(hex):match("#?(%x%x)(%x%x)(%x%x)")
  if not r then return Color{r=255,g=0,b=255,a=255} end
  return Color{ r=tonumber(r,16), g=tonumber(g,16), b=tonumber(b,16), a=255 }
end

local function safeClose(obj)
  if obj then pcall(function() obj:close() end) end
end

local function eqRGB(a,b) return a.red==b.red and a.green==b.green and a.blue==b.blue end

local function isImage(path)
  local ext = string.lower(fs.fileExtension(path) or ""):gsub("^%.","")
  return (ext=="png" or ext=="bmp" or ext=="gif" or ext=="ase" or ext=="aseprite"
       or ext=="jpg" or ext=="jpeg" or ext=="pcx" or ext=="tga" or ext=="webp")
end

local function listFilesRec(root, recursive, out)
  out = out or {}
  for _,name in ipairs(fs.listFiles(root)) do
    local p = fs.joinPath(root, name)
    if fs.isFile(p) then
      if isImage(p) then table.insert(out, p) end
    elseif recursive and fs.isDirectory(p) then
      listFilesRec(p, true, out)
    end
  end
  return out
end

local function ensureOutDir(basePath)
  local baseDir = fs.isDirectory(basePath) and basePath or fs.filePath(basePath)
  local out = fs.joinPath(baseDir, "out")
  if not fs.isDirectory(out) then fs.makeAllDirectories(out) end
  return out
end

local function toIndexed(sprite)
  if sprite.colorMode ~= ColorMode.INDEXED then
    app.activeSprite = sprite
    app.command.ChangePixelFormat{ format="indexed", dithering="none", ui=false }
  end
end

local function firstCelImage(spr)
  local ly = spr.layers[1]
  local cel = ly and ly:cel(1)
  return cel and cel.image or nil
end

-- Border scan (counts by index)
local function borderCounts(img)
  local w,h = img.width, img.height
  local freq = {}
  local function bump(v) freq[v]=(freq[v] or 0)+1 end
  for x=0,w-1 do bump(img:getPixel(x,0)); bump(img:getPixel(x,h-1)) end
  for y=0,h-1 do bump(img:getPixel(0,y)); bump(img:getPixel(w-1,y)) end
  return freq
end
local function majorityIndex(freq)
  local bestI,bestC=nil,-1
  for i,c in pairs(freq) do if c>bestC then bestC=c; bestI=i end end
  return bestI or 0
end

-- If manual/sample: prefer border index whose RGB == chosenColor; else majority index.
local function findBgIndex(spr, mode, chosenColor)
  local img = firstCelImage(spr); if not img then return 0 end
  local pal = spr.palettes[1]
  local freq = borderCounts(img)
  if mode=="manual" or mode=="sample" then
    local matchI,bestC=nil,-1
    for i,cnt in pairs(freq) do
      if eqRGB(pal:getColor(i), chosenColor) and cnt>bestC then matchI=i; bestC=cnt end
    end
    if matchI then return matchI end
  end
  return majorityIndex(freq)
end

-- apply a mapping oldIndex -> newIndex in one pass (no chaining)
local function applyIndexMapToSprite(spr, map)
  for _,ly in ipairs(spr.layers) do
    if ly.isImage then
      for _,cel in ipairs(ly.cels) do
        local img = cel.image
        for it in img:pixels() do
          local idx = it()
          local new = map[idx]
          if new ~= nil then it(new) end
        end
      end
    end
  end
end

-- rotate palette so bgIdx moves to 0; preserve relative order of others
local function rotatePalette(pal, bgIdx)
  if bgIdx == 0 then return end
  local n = #pal
  local old = {}
  for i=0,n-1 do old[i] = pal:getColor(i) end
  pal:setColor(0, old[bgIdx])
  for i=0,bgIdx-1 do pal:setColor(i+1, old[i]) end
  for i=bgIdx+1,n-1 do pal:setColor(i, old[i]) end
end

local function makeCanvasFrom(srcSpr, width, height)
  local dstSpr = Sprite(width, height, ColorMode.INDEXED)
  dstSpr:setPalette(srcSpr.palettes[1])
  local dstImg = dstSpr.layers[1]:cel(1).image
  for it in dstImg:pixels() do it(0) end -- fill with index 0
  local ox = math.floor((width - srcSpr.width)/2)
  local oy = math.floor((height - srcSpr.height)/2)
  local srcImg = firstCelImage(srcSpr)
  if srcImg then dstImg:drawImage(srcImg, Point(ox, oy)) end
  dstSpr.transparentColor = srcSpr.transparentColor
  return dstSpr
end

-- ---------- core: stable rotation + pixel unification ----------
local function forceBgIndex0_rotateKeepOrder(spr, mode, chosenColor, makeTrans0)
  toIndexed(spr)
  local pal = spr.palettes[1]

  if mode=="alpha" then
    spr.transparentColor = 0
    return
  end

  -- 1) decide BG index & color
  local bgIdx = findBgIndex(spr, mode, chosenColor)
  local bgRGB = (mode=="manual" or mode=="sample") and chosenColor or pal:getColor(bgIdx)

  -- 2) pixel index mapping = base rotation + unify any BG-colored slots -> 0
  local map = {}
  local n = #pal
  if bgIdx ~= 0 then
    -- base rotation mapping
    for i=0,n-1 do
      if     i == bgIdx then map[i] = 0
      elseif i <  bgIdx then map[i] = i+1
      else                  map[i] = i
      end
    end
  end
  -- unify duplicates of BG RGB to 0
  for i=0,n-1 do
    if i ~= bgIdx and eqRGB(pal:getColor(i), bgRGB) then map[i] = 0 end
  end
  if next(map) ~= nil then applyIndexMapToSprite(spr, map) end

  -- 3) rotate the PALETTE (order-preserving)
  rotatePalette(pal, bgIdx)
  -- ensure exact RGB at slot 0 (covers manual/sample)
  pal:setColor(0, bgRGB)
  spr:setPalette(pal)

  -- 4) transparency
  spr.transparentColor = makeTrans0 and 0 or -1
end

-- CONSOLIDATED HELPERS (PLACE BEFORE the UI / processOneFile)

local function readBytes(path, n, offset)
  local f, err = io.open(path, "rb")
  if not f then return nil, "open failed: "..tostring(err) end
  if offset and offset > 0 then f:seek("set", offset) end
  local data = f:read(n or 1024)
  f:close()
  return data
end

local function interpretRawEntries3(data, count)
  count = count or 16
  local out = {}
  for i = 0, math.min(count-1, 255) do
    local off = i*3
    if off+3 <= #data then
      out[#out+1] = string.format("[%d]=#%02X%02X%02X", i, data:byte(off+1), data:byte(off+2), data:byte(off+3))
    else break end
  end
  return table.concat(out, " ")
end

local function readPalFile(path)
  local f, err = io.open(path, "rb")
  if not f then return nil, "open failed: "..tostring(err) end
  local data = f:read("*all")
  f:close()
  return data
end

local function writePalFile(path, data)
  local f, err = io.open(path, "wb")
  if not f then return false, "open failed: "..tostring(err) end
  f:write(data)
  f:close()
  return true
end

local function paletteToTable(pal)
  local t = {}
  local n = (#pal>0) and #pal or 256
  for i=0,255 do
    if i < n then
      local ok, c = pcall(function() return pal:getColor(i) end)
      if ok and c then t[i] = { r = c.red or 0, g = c.green or 0, b = c.blue or 0 }
      else t[i] = { r=0,g=0,b=0 } end
    else
      t[i] = { r=0,g=0,b=0 }
    end
  end
  return t
end

local function dbgPrintPalTableFirstEntries(tbl, n)
  n = n or 8
  local s = {}
  for i=0, math.min(255, n-1) do
    local c = tbl[i] or { r=0,g=0,b=0 }
    s[#s+1] = string.format("[%d]=#%02X%02X%02X", i, c.r, c.g, c.b)
  end
  print("DBG: palette first entries -> "..table.concat(s, " "))
end

-- I/O & palette helpers (PLACE BEFORE processOneFile)

local function readFile(path)
  local f = io.open(path, "rb")
  if not f then return nil, "Can't open file: "..tostring(path) end
  local data = f:read("*all")
  f:close()
  return data
end

local function writeFile(path, data)
  local f = io.open(path, "wb")
  if not f then return false end
  f:write(data)
  f:close()
  return true
end

-- Write Adobe .ACT palette file (256 colors, 3 bytes per color)
local function writeActPalette(path, palTable)
  local f, err = io.open(path, "wb")
  if not f then return false, "Can't open file: "..tostring(err) end
  for i = 0, 255 do
    local c = palTable[i] or { r=0, g=0, b=0 }
    f:write(string.char(c.r or 0, c.g or 0, c.b or 0))
  end
  f:close()
  return true
end

-- Replace your current processOneFile with this robust version
local function processOneFile(path, width, height, mode, chosenColor, makeTrans0, outDir, headless)
  local spr = nil
  local outSpr = nil
  local savedSpr = nil
  local palTable = nil
  local outPath = nil

  -- open source sprite
  local ok, err = pcall(function() spr = Sprite{ fromFile = path } end)
  if not ok or not spr then
    if headless then print("Skip: "..tostring(path) .. " ("..tostring(err)..")") end
    return nil
  end

  -- ensure we always close on early return / error
  local status, res = pcall(function()
    app.activeSprite = spr
    pcall(function() app.command.FlattenLayers{ ui=false } end)

    forceBgIndex0_rotateKeepOrder(spr, mode, chosenColor, makeTrans0)

    outSpr = makeCanvasFrom(spr, width, height)
    outPath = fs.joinPath(outDir, fs.fileTitle(path) .. ".png")

    -- Save output PNG (may remap indices)
    outSpr:saveAs(outPath)

    -- Re-open saved PNG to read exact palette bytes that were actually written
    local ok2 = pcall(function() savedSpr = Sprite{ fromFile = outPath } end)
    if ok2 and savedSpr then
      if savedSpr.colorMode ~= ColorMode.INDEXED then
        savedSpr:convertColorMode(ColorMode.INDEXED)
      end
      local pal = savedSpr.palettes[1]
      palTable = {}
      for i = 0, 255 do
        local okc, c = pcall(function() return pal:getColor(i) end)
        if okc and c then palTable[i] = { r=c.red or 0, g=c.green or 0, b=c.blue or 0 }
        else palTable[i] = { r=0,g=0,b=0 } end
      end
      -- defensive ensure index 0 is taken from the saved palette
      local ok0, c0 = pcall(function() return pal:getColor(0) end)
      if ok0 and c0 then palTable[0] = { r=c0.red, g=c0.green, b=c0.blue } end
    else
      -- fallback: extract from outSpr (should exist)
      local pal = outSpr and outSpr.palettes and outSpr.palettes[1]
      palTable = {}
      if pal then
        for i = 0, 255 do
          local okc, c = pcall(function() return pal:getColor(i) end)
          if okc and c then palTable[i] = { r=c.red or 0, g=c.green or 0, b=c.blue or 0 }
          else palTable[i] = { r=0,g=0,b=0 } end
        end
      else
        for i=0,255 do palTable[i] = { r=0,g=0,b=0 } end
      end
    end
  end)

  -- always close sprites we opened
  safeClose(savedSpr)
  safeClose(outSpr)
  safeClose(spr)

  if not status then
    if headless then print("Error processing: "..tostring(res)) end
    return nil
  end

  if headless then print("Saved: "..tostring(outPath)) end
  return outPath, palTable
end

-- ---------- UI ----------

local function processFilesInList(fileList, width, height, mode, chosenColor, makeTrans0, outDir)
  local palTable = nil
  for _,inPath in ipairs(fileList) do
    local ok, outPath, palT = pcall(processOneFile, inPath, width, height, mode, chosenColor, makeTrans0, outDir, true)
    if ok then
      palTable = palT
    else
      print("ERR: "..tostring(outPath))
    end
  end
  return palTable
end

local isBatch = app.isUIAvailable == false or (app.params.input ~= nil)

if isBatch then
  local inPath = app.params.input
  local recursive = tonumber(app.params.recursive or "0") == 1
  local width = tonumber(app.params.width) or 256
  local height = tonumber(app.params.height) or 256
  local mode = (app.params.mode == "alpha") and "alpha" or "auto"
  local chosenColor = hexToColor(app.params.bg)
  local makeTrans0 = (app.params.trans0 == "1")
  local outDir = ensureOutDir(inPath)

  local fileList = fs.isDirectory(inPath) and listFilesRec(inPath, recursive) or { inPath }
  local palTable = processFilesInList(fileList, width, height, mode, chosenColor, makeTrans0, outDir)

  if palTable and #palTable > 0 then
    local actOutPath = fs.joinPath(outDir, "palette.act")
    local ok, err = writeActPalette(actOutPath, palTable)
    if not ok then print("ACT palette error: "..tostring(err)) end
  end
else
  -- UI mode: show dialog
  local state = { palColors={}, palLabels={} }
  local dlg = Dialog{ title="BG to Index 0" }

  dlg:combobox{ id="scope", label="Process", option="Single file", options={"Single file"},
    onchange=function()
      local asFolder = (dlg.data.scope=="Folder")
      dlg:modify{ id="file",       visible = not asFolder }
      dlg:modify{ id="folderAny",  visible =  asFolder }
      dlg:modify{ id="recursive",  visible =  asFolder }
    end
  }
  dlg:file{ id="file", label="File", open=true }
  dlg:check{ id="recursive", label="Include subfolders", selected=false, visible=false }

  dlg:separator{}
  dlg:combobox{ id="mode", label="Background", option="Auto (detect from borders)",
    options={"Auto (detect from borders)","Pick from sample palette","Pick color manually","No background (alpha)"},
    onchange=function()
      local m = dlg.data.mode
      dlg:modify{ id="manualColor",  visible = (m=="Pick color manually") }
      dlg:modify{ id="paletteColor", visible = (m=="Pick from sample palette") }
      dlg:modify{ id="load",         visible = (m=="Pick from sample palette") }
      dlg:modify{ id="trans0",       visible = (m~="No background (alpha)") }
    end
  }

  dlg:button{ id="load", text="Load palette (from 'File' or the folder's sample)", visible=false,
    onclick=function()
      local sample = dlg.data.file
      if (not sample or sample=="") and dlg.data.folderAny and dlg.data.folderAny~="" then sample = dlg.data.folderAny end
      if not sample or sample=="" then app.alert("Pick a sample file first."); return end
      local ok, spr = pcall(function() return Sprite{ fromFile=sample } end)
      if not ok or not spr then app.alert("Couldn’t open sample."); return end
      app.activeSprite = spr; toIndexed(spr)
      local pal = spr.palettes[1]
      state.palColors, state.palLabels = {}, {}
      for i=0,#pal-1 do
        local c = pal:getColor(i)
        table.insert(state.palColors, c)
        table.insert(state.palLabels, string.format("%3d  #%02x%02x%02x", i, c.red, c.green, c.blue))
      end
      spr:close()
      dlg:modify{ id="paletteColor", options=state.palLabels, visible=true }
    end
  }
  dlg:combobox{ id="paletteColor", options={}, visible=false }
  dlg:color{ id="manualColor", color=Color{ r=255,g=0,b=255,a=255 }, visible=false }
  dlg:check{ id="trans0", label="Make index 0 transparent", selected=false, visible=true }
  dlg:entry{ id="width", label="Width", text="256" }
  dlg:entry{ id="height", label="Height", text="256" }

  dlg:button{ text="Run", focus=true, onclick=function()
    local d = dlg.data

    print("DBG: dlg.file =", tostring(d.file))

    local inPath = d.file
    if not inPath or inPath=="" then
      app.alert("Pick a file.")
      return
    end

    local width = tonumber(d.width or "256") or 256
    local height = tonumber(d.height or "256") or 256
    local mode = (d.mode=="No background (alpha)") and "alpha"
              or (d.mode=="Pick from sample palette") and "sample"
              or (d.mode=="Pick color manually") and "manual"
              or "auto"

    local bgColor = (mode=="manual") and d.manualColor or
                    (mode=="sample" and state.palColors[ (type(d.paletteColor)=="number" and d.paletteColor) or ((tonumber((d.paletteColor or ""):match("^%s*(%d+)")) or 0)+1) ]) or
                    Color{r=255,g=0,b=255,a=255}

    local outPath, palTable = processOneFile(inPath, width, height, mode, bgColor, d.trans0, ensureOutDir(inPath), false)

    local actOutPath = outPath:gsub("%.png$", ".act")
    local okAct, errAct = writeActPalette(actOutPath, palTable)
    if okAct then
      app.alert("Wrote ACT palette:\n" .. actOutPath)
    else
      app.alert("Failed to write ACT palette:\n" .. tostring(errAct))
    end
  end}
  dlg:button{ text="Cancel" }

  dlg:show()
end
