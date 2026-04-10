.PHONY: install install-all dev clean clean-cache clean-all \
       setup-theta theta-start theta-stop theta-status \
       calibrate calibrate-expanded \
       scan scan-detail plan \
       portfolio portfolio-greeks portfolio-conservative portfolio-log \
       backtest-bps backtest-ic backtest-all \
       validate theta-validate \
       train-ml test

PYTHON := python
PIP := pip
JAVA := /usr/lib/jvm/java-21-openjdk-amd64/bin/java
THETA_JAR := $(HOME)/Downloads/ThetaTerminalv3.jar
THETA_URL := https://download-unstable.thetadata.us/ThetaTerminalv3.jar
THETA_CREDS := creds.txt
THETA_PID := /tmp/theta-terminal.pid

# ── Setup ──────────────────────────────────────────────

install:
	$(PIP) install -e .

dev:
	$(PIP) install -e ".[dev,validate]"

install-all: install-java install-theta install-deps
	@echo "All dependencies installed."

install-deps:
	$(PIP) install -e ".[dev,validate]"
	$(PIP) install scikit-learn httpx

install-java:
	@which $(JAVA) > /dev/null 2>&1 || { \
		echo "Installing Java 21..."; \
		sudo apt-get update -qq && sudo apt-get install -y -qq openjdk-21-jre-headless; \
	}
	@$(JAVA) -version 2>&1 | head -1

install-theta:
	@if [ ! -f "$(THETA_JAR)" ]; then \
		echo "Downloading Theta Terminal v3..."; \
		curl -L -o "$(THETA_JAR)" "$(THETA_URL)"; \
		echo "Downloaded to $(THETA_JAR)"; \
	else \
		echo "Theta Terminal already at $(THETA_JAR)"; \
	fi

setup-creds:
	@if [ ! -f "$(THETA_CREDS)" ]; then \
		echo "Creating credentials file..."; \
		read -p "ThetaData email: " email; \
		read -sp "ThetaData password: " pass; echo; \
		echo "$$email" > $(THETA_CREDS); \
		echo "$$pass" >> $(THETA_CREDS); \
		chmod 600 $(THETA_CREDS); \
		echo "Credentials saved to $(THETA_CREDS)"; \
	else \
		echo "Credentials file exists at $(THETA_CREDS)"; \
	fi

# ── ThetaData Terminal ─────────────────────────────────

theta-start: install-theta
	@if [ -f "$(THETA_PID)" ] && kill -0 $$(cat $(THETA_PID)) 2>/dev/null; then \
		echo "Theta Terminal already running (PID $$(cat $(THETA_PID)))"; \
	else \
		echo "Starting Theta Terminal v3..."; \
		$(JAVA) -jar $(THETA_JAR) --creds-file $(THETA_CREDS) & echo $$! > $(THETA_PID); \
		sleep 5; \
		$(PYTHON) -c "from weeklyopt.validation import ThetaDataClient; c=ThetaDataClient(); print('Connected:', c.check_connection())"; \
	fi

theta-stop:
	@if [ -f "$(THETA_PID)" ] && kill -0 $$(cat $(THETA_PID)) 2>/dev/null; then \
		kill $$(cat $(THETA_PID)); \
		rm -f $(THETA_PID); \
		echo "Theta Terminal stopped"; \
	else \
		pkill -f ThetaTerminalv3 2>/dev/null || true; \
		rm -f $(THETA_PID); \
		echo "Not running"; \
	fi

theta-status:
	@if [ -f "$(THETA_PID)" ] && kill -0 $$(cat $(THETA_PID)) 2>/dev/null; then \
		echo "Theta Terminal running (PID $$(cat $(THETA_PID)))"; \
		$(PYTHON) -c "from weeklyopt.validation import ThetaDataClient; c=ThetaDataClient(); print('API connected:', c.check_connection())"; \
	else \
		echo "Theta Terminal not running"; \
	fi

# ── Clean ─────────────────────────────────────────────

clean:
	rm -rf build dist *.egg-info weeklyopt.egg-info
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete

clean-cache:
	rm -rf $(HOME)/.weeklyopt_cache

clean-all: clean clean-cache
	rm -f $(THETA_PID)

# ── Calibration ────────────────────────────────────────

calibrate:
	weeklyopt calibrate --weeks 8

calibrate-expanded:
	weeklyopt calibrate --tickers SPY QQQ AAPL MSFT AMZN TSLA NVDA META GOOGL AMD JPM BAC XLF IWM GLD TLT XLE XBI KRE COIN SLV EEM SMH --weeks 8

# ── Weekly Workflow ────────────────────────────────────

scan:
	weeklyopt scan --expanded --credit-spreads

scan-detail:
	weeklyopt scan --expanded --credit-spreads --detail

plan:
	weeklyopt scan --expanded --credit-spreads --trade-plan --capital 20000 --risk-limit 0.10

# ── Portfolio Backtests ────────────────────────────────

portfolio:
	weeklyopt portfolio --capital 20000 --risk-limit 0.10 --max-contracts 10 --credit-spreads --expanded --exit-rules greeks --min-score 50

portfolio-greeks:
	weeklyopt portfolio --capital 20000 --risk-limit 0.10 --max-contracts 10 --credit-spreads --expanded --exit-rules greeks --min-score 50 --weekly-log 20

portfolio-conservative:
	weeklyopt portfolio --capital 20000 --risk-limit 0.05 --max-contracts 5 --credit-spreads --expanded --exit-rules conservative --min-score 50

portfolio-log:
	weeklyopt portfolio --capital 20000 --risk-limit 0.10 --max-contracts 10 --credit-spreads --expanded --exit-rules greeks --min-score 50 --weekly-log 20

# ── Single Strategy Backtests ──────────────────────────

backtest-bps:
	weeklyopt backtest --strategy bull_put_spread --exit-rules greeks --no-plot

backtest-ic:
	weeklyopt backtest --strategy iron_condor --exit-rules greeks --no-plot

backtest-all:
	@for strat in bull_put_spread bear_call_spread iron_condor; do \
		echo "=== $$strat ===" ; \
		weeklyopt backtest --strategy $$strat --start 2023-01-01 --end 2025-12-31 --exit-rules greeks --no-plot ; \
		echo "" ; \
	done

# ── Validation ─────────────────────────────────────────

validate:
	weeklyopt validate --tickers SPY NVDA AAPL --delta 0.30 --type put

theta-validate:
	weeklyopt theta-validate --strategy bull_put_spread --tickers SPY NVDA --start 2025-06-01 --end 2025-12-31 --max-trades 10

# ── ML Model ──────────────────────────────────────────

train-ml:
	$(PYTHON) -m weeklyopt.train_ml_exit_v2

# ── Tests ─────────────────────────────────────────────

test:
	@$(PYTHON) -c "from weeklyopt.pricing.black_scholes import bs_price, OptionType; p=bs_price(500,500,5/252,0.045,0.25,OptionType.CALL); assert 5 < p < 10; print('Pricing OK')"
	@$(PYTHON) -c "from weeklyopt.strategies import BullPutSpread; s=BullPutSpread(); legs=s.construct(500,0.25,5/252,0.045); assert len(legs)==2; print('Strategies OK')"
	@$(PYTHON) -c "from weeklyopt.pricing.calibration import IVCalibrator; c=IVCalibrator.load_calibration(); print(f'Calibration OK: {len(c)} tickers' if c else 'No calibration cached')"
	@$(PYTHON) -c "from weeklyopt.engine.ml_exit import MLExitModel; m=MLExitModel.load(); print(f'ML model: {\"trained\" if m.is_trained else \"not trained\"}')"
	@echo "All checks passed."
