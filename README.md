# Indian Insider → moved to StockLens

This repository has been **merged into** [StockLens](https://github.com/Karthik96CFA/stocklens) as the `engine/` directory.

Use the monorepo going forward:

```powershell
git clone https://github.com/Karthik96CFA/stocklens.git
cd stocklens
powershell -ExecutionPolicy Bypass -File engine\install\setup_windows.ps1
```

Copy your existing `.env` and `.state/` folder to `engine/` in the StockLens clone.

See [stocklens/engine/README.md](https://github.com/Karthik96CFA/stocklens/blob/master/engine/README.md) and [DEPLOYMENT.md](https://github.com/Karthik96CFA/stocklens/blob/master/DEPLOYMENT.md) for full setup.

This repo remains available for history; new development happens in **stocklens**.
