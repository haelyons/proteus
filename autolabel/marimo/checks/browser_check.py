"""Exercise an already running local marimo editor via Chrome's debugging endpoint."""

import asyncio
import base64
import json
import urllib.request
from pathlib import Path

import websockets

HERE = Path(__file__).resolve().parent
DEEP = """window.deep = (sel, root=document) => [...root.querySelectorAll(sel),
    ...[...root.querySelectorAll('*')].flatMap(x => x.shadowRoot ? deep(sel,x.shadowRoot) : [])];"""


async def main():
    pages = json.load(urllib.request.urlopen("http://127.0.0.1:9223/json"))
    page = next(p for p in pages if p["type"] == "page")
    async with websockets.connect(page["webSocketDebuggerUrl"], max_size=20_000_000) as ws:
        request_id = 0

        async def command(method, params):
            nonlocal request_id
            request_id += 1
            await ws.send(json.dumps({"id": request_id, "method": method, "params": params}))
            while True:
                result = json.loads(await ws.recv())
                if result.get("id") == request_id:
                    if "error" in result:
                        raise RuntimeError(result)
                    return result["result"]

        async def evaluate(expression):
            result = await command("Runtime.evaluate", {"expression": DEEP + expression, "returnByValue": True})
            if "exceptionDetails" in result:
                raise RuntimeError(result)
            return result["result"].get("value")

        async def wait_for(expression):
            for _ in range(80):
                if await evaluate(expression):
                    return
                await asyncio.sleep(0.25)
            raise AssertionError("Browser condition timed out: " + expression)

        # Labels live in nested shadow roots. Inputs are populated through native React setters.
        async def fill_prediction(unit):
            await evaluate("""
            (() => {
              const e = deep('input').find(x=>x.dataset.testid==='marimo-plugin-text-input');
              if (!e) throw new Error('reader input missing');
              e.focus();
              Object.getOwnPropertyDescriptor(HTMLInputElement.prototype,'value').set.call(e,'AUTOMATED BROWSER CHECK');
              e.dispatchEvent(new Event('input',{bubbles:true,composed:true})); e.blur();
              deep('textarea').forEach((x,i)=>{
                x.focus();
                Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype,'value').set.call(x,
                    ['Prior reference already known; UI test only','Orientation sensitivity','Janus cached views','Invariant response'][i]);
                x.dispatchEvent(new Event('input',{bubbles:true,composed:true})); x.blur();
              });
              const category=deep('select').find(x=>x.options[0].value.startsWith('low ('));
              category.value='low (≤20)'; category.dispatchEvent(new Event('change',{bubbles:true}));
            })();""")
            if unit == "mixed4e:55":
                await evaluate("""(() => {const x=deep('select').find(x=>x.options[0].value==='grayscale');
                x.value='rotate_180';x.dispatchEvent(new Event('change',{bubbles:true}));})();""")
            await asyncio.sleep(0.6)

        await wait_for("deep('button').some(x=>x.innerText==='Commit prediction and reveal')")
        results = []
        for unit in ("mixed4b:431", "mixed4e:55"):
            if unit == "mixed4e:55":
                await evaluate("""(() => {const x=deep('select').find(x=>x.options[0].value==='mixed4b:431');
                x.value='mixed4e:55';x.dispatchEvent(new Event('change',{bubbles:true}));})();""")
                await wait_for("[...document.querySelectorAll('.output-area')].some(x=>x.innerText.includes('mixed4e:55 · measurement'))")
                await wait_for("deep('button').some(x=>x.innerText==='Commit prediction and reveal')")
            assert await evaluate("[...document.querySelectorAll('.output-area')].some(x=>x.innerText.includes('Results hidden.'))")
            assert not await evaluate("[...document.querySelectorAll('.output-area')].some(x=>x.innerText.includes('Reveal · cached measurements'))")
            await fill_prediction(unit)
            await evaluate("deep('button').find(x=>x.innerText==='Commit prediction and reveal').click()")
            await wait_for("[...document.querySelectorAll('.output-area')].some(x=>x.innerText.includes('Reveal · cached measurements'))")
            assert not await evaluate("[...document.querySelectorAll('.output-area')].some(x=>x.innerText.includes('Save failed:'))")
            await wait_for("deep('button').some(x=>x.innerText==='Save revised reading')")
            await evaluate("""deep('textarea').forEach(x=>{
              x.focus();
              Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype,'value').set.call(x,'AUTOMATED UI CHECK; not a scientific reading');
              x.dispatchEvent(new Event('input',{bubbles:true,composed:true}));x.blur();
            });""")
            await asyncio.sleep(0.6)
            await evaluate("deep('button').find(x=>x.innerText==='Save revised reading').click()")
            await wait_for("[...document.querySelectorAll('.output-area')].some(x=>x.innerText.includes('Saved:'))")
            # Open checks to exercise the widest table, then capture the rendered evidence.
            await evaluate("deep('button').filter(x=>x.getAttribute('role')==='tab' && x.innerText==='Every-pair checks').forEach(x=>x.click())")
            await command("Emulation.setDeviceMetricsOverride", {"width": 1440, "height": 1000, "deviceScaleFactor": 1, "mobile": False})
            await evaluate("document.querySelectorAll('.output-area')[0].scrollIntoView()")
            shot = await command("Page.captureScreenshot", {"format": "png"})
            (HERE / f"{unit.replace(':', '_')}.png").write_bytes(base64.b64decode(shot["data"]))
            results.append({"unit": unit, "hidden_before_commit": True, "reveal": True, "revision_saved": True})
        (HERE / "browser_results.json").write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(results))


if __name__ == "__main__":
    asyncio.run(main())
