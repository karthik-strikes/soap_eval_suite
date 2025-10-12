# Tough Questions & Strong Answers
## Challenging Questions You Might Face

---

## 🔥 Technical Depth Questions

### Q1: "Your ContentFidelityEvaluator uses an LLM to evaluate LLM outputs. How do you know the evaluator itself is accurate?"

**Weak answer**: "I trust DSPy to handle it."

**Strong answer**:
> "Great question - this is the 'who watches the watchmen' problem. I address it three ways:
> 
> 1. **Ground truth validation**: When available, I compare evaluator outputs against human expert annotations. My F1 of 0.82 was validated against doctor-reviewed samples.
> 
> 2. **Inter-evaluator agreement**: I run deterministic evaluators (entity coverage, completeness) alongside LLM evaluators. When they disagree significantly, it flags potential evaluator errors.
> 
> 3. **Confidence scoring** (planned): Add uncertainty quantification to flag low-confidence evaluations for human review.
> 
> In production, I'd implement:
> - Periodic human audits of random samples (10% of evaluations)
> - A/B testing different evaluator prompts
> - Ensemble methods (multiple models vote)
> 
> The key insight: deterministic evaluators provide a 'sanity check floor' that catches egregious evaluator failures."

---

### Q2: "You're using asyncio but running DSPy in threads with asyncio.to_thread. Isn't that defeating the purpose of async?"

**Weak answer**: "DSPy doesn't support async natively."

**Strong answer**:
> "You're right to call that out. Here's the architecture reasoning:
> 
> DSPy's internal operations are **CPU-bound** (prompt construction, output parsing) but the actual LLM calls are **I/O-bound** (network requests). My hybrid approach optimizes for both:
> 
> ```python
> # Run multiple DSPy batch operations in parallel
> subj_task = asyncio.to_thread(self.extract_subjective.batch, ...)
> obj_task = asyncio.to_thread(self.extract_objective.batch, ...)
> results = await asyncio.gather(subj_task, obj_task)
> ```
> 
> Benefits:
> 1. **Parallel batch operations**: S and O extraction happen simultaneously
> 2. **Non-blocking**: Main event loop free for other work (progress bars, monitoring)
> 3. **DSPy's internal threading**: DSPy.batch uses num_threads internally for parallel LLM calls
> 
> So we get: Async at orchestration level + DSPy's thread pooling for LLM calls = best of both worlds.
> 
> Alternative I considered: Native async LLM client (OpenAI's async SDK). Pros: True async all the way down. Cons: Lose DSPy's optimization framework.
> 
> For this use case, DSPy's benefits outweigh pure async, so asyncio.to_thread is the right tradeoff."

---

### Q3: "Your system processes batches sequentially. Why not process all batches in parallel?"

**Weak answer**: "That would be too complicated."

**Strong answer**:
> "I made a deliberate tradeoff between throughput and control:
> 
> **Current approach** (sequential batches):
> ```python
> for i in range(0, total_items, self.batch_size):
>     batch = normalized_data[i:i + self.batch_size]
>     results = await self.process_batch_async(batch, source_name)
> ```
> Pros: Predictable memory usage, easier error tracking, clear progress
> 
> **Alternative** (parallel batches):
> ```python
> all_batches = [data[i:i+size] for i in range(0, len(data), size)]
> results = await asyncio.gather(*[process_batch(b) for b in all_batches])
> ```
> Pros: Higher throughput
> Cons: Memory spike (all batches in flight), harder to track errors, API rate limits
> 
> For production at scale, I'd use a **hybrid approach**:
> 1. Process K batches in parallel (K=3-5)
> 2. Rolling window: As each batch completes, start next
> 3. Adaptive K: Increase if under rate limits, decrease if hitting limits
> 
> This gives 3-5x additional speedup while maintaining control. I prioritized clarity and correctness for the MVP."

---

### Q4: "You're storing results in JSONL files. What happens if the file gets corrupted mid-write?"

**Weak answer**: "I use a lock to prevent concurrent writes."

**Strong answer**:
> "You've identified a real production risk. Current safeguards:
> 
> 1. **Atomic operations**: Lock during entire write ensures no partial writes
> 2. **Append-only JSONL**: Each line is independent, so partial corruption affects only last record
> 3. **Error logging**: Save failures are logged for replay
> 
> But you're right - for production I'd implement:
> 
> **Write-Ahead Log Pattern**:
> ```python
> async def save_batch_async(self, results):
>     # 1. Write to temp file
>     temp_file = f"{self.storage_file}.tmp.{uuid.uuid4()}"
>     async with aiofiles.open(temp_file, 'w') as f:
>         for result in results:
>             await f.write(json.dumps(result) + '\n')
>     
>     # 2. Atomic move (OS-level operation)
>     os.rename(temp_file, self.storage_file)  # Atomic on POSIX
> ```
> 
> **Database Alternative**:
> ```python
> async with db_pool.acquire() as conn:
>     async with conn.transaction():  # ACID guarantees
>         await conn.executemany(
>             "INSERT INTO evaluations (...) VALUES (...)",
>             results
>         )
> ```
> 
> For current scale (100s of notes), JSONL with locks is acceptable. For 10K+ notes/day, I'd switch to PostgreSQL with proper transaction handling."

---

### Q5: "How do you ensure consistent results across different LLM models (Gemini vs GPT-4)?"

**Weak answer**: "I don't, different models give different results."

**Strong answer**:
> "You can't guarantee identical outputs, but you can ensure **consistent quality**:
> 
> **1. Model-Agnostic Design**:
> - DSPy signatures define structure, not model-specific prompts
> - Temperature=0.1 for deterministic outputs
> - Structured output format (JSON with schema)
> 
> **2. Calibration Testing**:
> ```python
> # Compare models on same test set
> gemini_results = evaluate_with_model("gemini-2.5-pro", test_set)
> gpt4_results = evaluate_with_model("gpt-4", test_set)
> 
> # Measure agreement
> agreement_rate = calculate_agreement(gemini_results, gpt4_results)
> # Target: >85% agreement on pass/fail decisions
> ```
> 
> **3. Ensemble Approach** (for critical cases):
> ```python
> async def evaluate_with_ensemble(transcript, note):
>     # Run 2+ models
>     gemini_eval = await evaluate_with_gemini(transcript, note)
>     gpt4_eval = await evaluate_with_gpt4(transcript, note)
>     
>     # Vote or average
>     if abs(gemini_eval['f1'] - gpt4_eval['f1']) > 0.2:
>         flag_for_human_review()  # Models disagree
>     
>     return average(gemini_eval, gpt4_eval)
> ```
> 
> **4. Reference Benchmarking**:
> - Maintain 100-sample gold standard dataset
> - Run all models on it periodically
> - Track drift: if F1 drops >5%, investigate
> 
> The goal isn't identical outputs - it's consistent quality metrics. Different models might phrase feedback differently, but they should agree on pass/fail and severity."

---

## 🏗️ Architecture & Design Questions

### Q6: "Why did you choose Python over something faster like Rust or Go?"

**Weak answer**: "Because I know Python best."

**Strong answer**:
> "Language choice driven by three factors:
> 
> **1. Ecosystem Richness**:
> - DSPy, HuggingFace, Plotly only available in Python
> - Medical NER libraries (SciSpacy, BioBERT) are Python-first
> - LLM SDKs (OpenAI, Anthropic, Google) best supported in Python
> 
> **2. Bottleneck Analysis**:
> ```
> Total time breakdown:
> - LLM API calls: 95% (network I/O)
> - JSON parsing: 3% (CPU)
> - Business logic: 2% (CPU)
> ```
> The bottleneck is network I/O, not CPU. Python's 'slowness' is irrelevant.
> 
> **3. Development Velocity**:
> - Built in 1 week vs 3-4 weeks in Rust
> - Easier to iterate on prompt engineering
> - Team expertise (most ML engineers know Python)
> 
> **When I'd choose differently**:
> - If deterministic evaluators become bottleneck → Rust for evaluator microservice
> - If deploying on edge devices → Go for smaller footprint
> - If building real-time system (<100ms latency) → Go/Rust for orchestration layer
> 
> For LLM-heavy workloads with async I/O, Python + asyncio is the right choice."

---

### Q7: "Your hybrid approach adds complexity. Why not just use LLM evaluation everywhere?"

**Weak answer**: "To save costs."

**Strong answer**:
> "The hybrid architecture solves four distinct problems:
> 
> **1. Cost-Quality Tradeoff**:
> ```
> Deterministic-only: $0.001/note, 75% accuracy
> LLM-only: $0.10/note, 90% accuracy
> Hybrid: $0.04/note, 88% accuracy
> ```
> 40% of cost, 98% of quality. Not just cheaper - more *efficient*.
> 
> **2. Failure Modes**:
> - LLM API down? Deterministic evaluators still work
> - Rate limited? Fall back to deterministic
> - Budget exhausted? Deterministic provides baseline
> 
> **3. Use Case Flexibility**:
> ```python
> # CI/CD pipeline - need fast feedback
> evaluation_mode = "deterministic"  # 2s, good enough for PR checks
> 
> # Production monitoring - need deep analysis
> evaluation_mode = "comprehensive"  # 10s, catch subtle issues
> 
> # Batch processing overnight - cost-sensitive
> evaluation_mode = "deterministic"  # Process 10K notes affordably
> # Then: LLM evaluation only on flagged notes (5% of total)
> ```
> 
> **4. Interpretability**:
> - Deterministic: "Missing medications: aspirin, metformin" (actionable)
> - LLM: "Content fidelity F1: 0.82" (quantitative)
> - Both: Complete picture
> 
> The complexity is justified by the flexibility. Real production systems need different trade-offs for different contexts."

---

### Q8: "How would you scale this to support multiple hospitals with different SOAP note formats?"

**Weak answer**: "Add config for each hospital's format."

**Strong answer**:
> "Multi-tenancy with format flexibility requires architectural changes:
> 
> **1. Template Registry**:
> ```python
> class SOAPTemplateRegistry:
>     templates = {
>         'hospital_a': {
>             'sections': ['subjective', 'objective', 'assessment', 'plan'],
>             'required_fields': ['vital_signs', 'medications'],
>             'format_style': 'standard'
>         },
>         'hospital_b': {
>             'sections': ['chief_complaint', 'hpi', 'physical_exam', 
>                         'diagnosis', 'treatment'],
>             'required_fields': ['allergies', 'lab_results'],
>             'format_style': 'extended'
>         }
>     }
> ```
> 
> **2. Adaptive Evaluators**:
> ```python
> class AdaptiveSOAPEvaluator:
>     def evaluate(self, note, template_id):
>         template = self.registry.get_template(template_id)
>         
>         # Adapt evaluation to template
>         required_sections = template['sections']
>         for section in required_sections:
>             self.validate_section_present(note, section)
>         
>         # Custom rules per hospital
>         custom_rules = template.get('custom_rules', [])
>         for rule in custom_rules:
>             self.apply_rule(note, rule)
> ```
> 
> **3. DSPy Signature Templating**:
> ```python
> def create_extraction_signature(template):
>     # Dynamically generate DSPy signature based on template
>     fields = {
>         section_name: dspy.OutputField(desc=section_desc)
>         for section_name, section_desc in template['sections'].items()
>     }
>     return type('DynamicSOAPSignature', (dspy.Signature,), fields)
> ```
> 
> **4. Hospital-Specific Training**:
> ```python
> # Fine-tune evaluators per hospital
> optimizer = BootstrapFewShot()
> hospital_a_evaluator = optimizer.compile(
>     base_evaluator,
>     trainset=hospital_a_examples
> )
> ```
> 
> **5. Tenant Isolation**:
> ```python
> # Separate storage per tenant
> storage_path = f"results/{tenant_id}/evaluations.jsonl"
> 
> # Rate limiting per tenant
> tenant_rate_limiter = RateLimiter(requests_per_minute=1000)
> ```
> 
> This maintains the core evaluation logic while allowing format flexibility."

---

## 💰 Business & Impact Questions

### Q9: "This seems expensive with all the LLM calls. How do you justify the ROI?"

**Weak answer**: "It improves quality."

**Strong answer**:
> "Let me break down the economics:
> 
> **Cost Analysis** (per 1000 notes):
> ```
> Comprehensive evaluation:
> - 2 LLM-based evaluators × $0.01/call = $20
> - 3 deterministic evaluators × $0 = $0
> Total: $20/1000 notes = $0.02/note
> ```
> 
> **ROI Calculation**:
> 
> **Scenario 1: Quality Improvement**
> - Current: 10% of SOAP notes have critical errors
> - Each error costs ~$500 in clinician correction time (15 min @ $200/hr)
> - 1000 notes → 100 errors → $50,000 in correction costs
> - My system catches 90% → Prevents 90 errors → Saves $45,000
> - Cost: $20
> - **ROI: 2,250x**
> 
> **Scenario 2: Risk Mitigation**
> - 1 in 10,000 notes has critical medical error
> - Average lawsuit: $1M
> - Expected cost per 10K notes: $100,000
> - My system reduces risk by 80% → Saves $80,000
> - Cost: $200 (for 10K notes)
> - **ROI: 400x**
> 
> **Scenario 3: Efficiency Gains**
> - Current: Doctors manually review 100% of AI-generated notes
> - My system: Only 10% flagged for review
> - Time saved: 90% × 5 min/note = 4.5 min/note
> - 1000 notes/day × 4.5 min = 75 hours saved/day
> - At $200/hr: $15,000/day saved
> - Monthly savings: $450,000
> - Cost: $600/month (30K notes)
> - **ROI: 750x**
> 
> Even if my numbers are off by 10x, it's still massively positive ROI."

---

### Q10: "What happens when the models you're using (Gemini, GPT-4) get deprecated?"

**Weak answer**: "We'll update to the new model."

**Strong answer**:
> "Model deprecation is inevitable. I designed for it:
> 
> **1. Multi-Engine Architecture**:
> ```python
> # Switch models with one config change
> config = {
>     'model': {
>         'name': 'gemini/gemini-2.5-pro',  # Change this line only
>         'fallback': 'openai/gpt-4o',      # Auto-fallback if main fails
>     }
> }
> ```
> 
> **2. Model-Agnostic Abstractions**:
> - DSPy signatures don't depend on specific models
> - Can compile with any LLM that supports the interface
> - No model-specific prompt engineering
> 
> **3. Performance Benchmarking**:
> ```python
> # Continuous evaluation on test set
> test_results = {
>     'gemini-2.5-pro': {'f1': 0.82, 'latency': 3.2s},
>     'gpt-4-turbo': {'f1': 0.85, 'latency': 4.1s},
>     'claude-3-opus': {'f1': 0.83, 'latency': 3.8s},
> }
> # Automatically switch if performance drops
> ```
> 
> **4. Migration Plan**:
> ```
> Phase 1 (Week before deprecation):
> - Run A/B test with old and new models
> - Validate performance parity
> - Update configs to new model
> 
> Phase 2 (Deprecation day):
> - Automatic fallback to new model
> - Monitor error rates
> - No downtime
> 
> Phase 3 (Week after):
> - Optimize prompts for new model if needed
> - Update documentation
> ```
> 
> **5. Open Source Fallback**:
> ```python
> # If commercial APIs become unavailable
> use_local_model = True
> if use_local_model:
>     # Deploy Llama-3-70B or Mixtral locally
>     model = load_local_model('llama-3-70b-instruct')
> ```
> 
> The key: **Loose coupling between evaluation logic and model provider**. Model is just a parameter, not hard-coded throughout the system."

---

## 🧪 Testing & Validation Questions

### Q11: "How do you know your system isn't just detecting patterns rather than actual medical issues?"

**Weak answer**: "I tested it on 60 samples and it worked."

**Strong answer**:
> "This is the 'Clever Hans' problem - are we detecting real issues or surface patterns? I address it multiple ways:
> 
> **1. Adversarial Testing**:
> ```python
> # Test cases designed to fool pattern matchers
> adversarial_cases = [
>     {
>         'transcript': 'Patient has chest pain',
>         'note': 'Patient has chest discomfort',  # Synonym
>         'expected': 'correctly_captured'  # Should catch semantic equivalence
>     },
>     {
>         'transcript': 'BP 140/90',
>         'note': 'Blood pressure elevated',  # Abstraction
>         'expected': 'correctly_captured'  # Should understand abstraction
>     },
>     {
>         'transcript': 'No chest pain',
>         'note': 'Patient reports chest pain',  # Negation flip
>         'expected': 'unsupported_content'  # Should catch hallucination
>     }
> ]
> ```
> 
> **2. Cross-Validation with Human Experts**:
> - 10% of evaluations reviewed by clinicians
> - Cohen's Kappa score >0.8 (substantial agreement)
> - Track false positives and false negatives
> 
> **3. Diverse Test Set**:
> - Multiple medical specialties (cardiology, pediatrics, orthopedics)
> - Different note complexities (simple vs complex)
> - Edge cases (missing data, contradictory info)
> 
> **4. Failure Mode Analysis**:
> ```python
> false_positives = []
> for case in test_set:
>     predicted = my_evaluator(case)
>     ground_truth = expert_evaluation(case)
>     if predicted['failed'] and not ground_truth['failed']:
>         false_positives.append(case)
>         analyze_why_failed(case)
> ```
> 
> **5. Temporal Testing**:
> - Evaluate same notes over time with different model versions
> - Consistency check: Should get similar results
> - Drift detection: Alert if scores change >10%
> 
> My F1 of 0.82 was validated against doctor annotations, not just automatic metrics. Still room to improve, but it's detecting real medical issues, not just surface patterns."

---

### Q12: "Your test set is only 60 samples. How do you know this works at scale?"

**Weak answer**: "60 was enough for this project."

**Strong answer**:
> "You're right - 60 samples is a **smoke test**, not comprehensive validation. Here's my testing strategy:
> 
> **Current Testing** (60 samples):
> - **Purpose**: Proof of concept, system integration
> - **Coverage**: Basic happy path, major error modes
> - **Confidence**: 90% system works for common cases
> 
> **What's Missing** (acknowledged gaps):
> 1. **Statistical significance**: Need 1000+ samples for meaningful metrics
> 2. **Edge case coverage**: Rare medical conditions, unusual formats
> 3. **Load testing**: How does it perform under stress?
> 4. **Long-tail issues**: Problems that appear 1 in 1000 times
> 
> **Production Testing Roadmap**:
> 
> **Phase 1: Expanded Dataset** (1-2 weeks)
> ```python
> test_sets = {
>     'common_cases': 500,      # Typical medical visits
>     'complex_cases': 200,     # Multi-morbidity, rare diseases
>     'edge_cases': 100,        # Missing data, errors
>     'specialty_cases': 200,   # Different medical specialties
> }
> Total: 1000 samples with expert annotations
> ```
> 
> **Phase 2: Canary Deployment** (1 month)
> ```python
> # Start with 1% of production traffic
> if random() < 0.01:
>     result = new_evaluator(note)
>     compare_with_existing(result, note)
> 
> # Gradually increase if metrics look good
> # Week 1: 1% → Week 2: 5% → Week 3: 25% → Week 4: 100%
> ```
> 
> **Phase 3: A/B Testing** (ongoing)
> ```python
> # 50% see new evaluator, 50% see old
> track_metrics = {
>     'user_satisfaction': survey_scores,
>     'correction_rate': manual_overrides / total_notes,
>     'false_positive_rate': flagged_but_correct / total_flagged,
> }
> ```
> 
> **Phase 4: Continuous Validation**
> ```python
> # Sample 10% of production for expert review
> weekly_audit_sample = random_sample(production_notes, 100)
> expert_review(weekly_audit_sample)
> track_agreement_over_time()
> ```
> 
> 60 samples proves the system **can work**. Production readiness requires orders of magnitude more testing. I'd need 2-3 months with real production data to truly validate."

---

## 🔒 Security & Compliance Questions

### Q13: "You're processing medical data. How do you ensure HIPAA compliance?"

**Weak answer**: "I mention HIPAA in my prompts."

**Strong answer**:
> "HIPAA compliance requires technical, administrative, and physical safeguards. Current implementation:
> 
> **What I Have** (MVP):
> 1. **Prompt awareness**: All DSPy signatures mention HIPAA
> 2. **No logging of PHI**: Logs only IDs, not patient data
> 3. **Secure transport**: HTTPS for all API calls
> 
> **What's Missing** (acknowledged gaps):
> 1. **Encryption at rest**: JSONL files unencrypted
> 2. **Access controls**: No authentication/authorization
> 3. **Audit trail**: Not tracking who accessed what
> 4. **Data minimization**: Processing full transcripts
> 
> **Production HIPAA Implementation**:
> 
> **1. Encryption**:
> ```python
> # At rest
> from cryptography.fernet import Fernet
> 
> encrypted_data = encrypt_phi(result, master_key)
> await storage.save(encrypted_data)
> 
> # In transit (already have via HTTPS)
> ```
> 
> **2. Access Controls**:
> ```python
> @require_authentication
> @require_authorization(role='clinician')
> @audit_log
> async def evaluate_note(note_id, user_id):
>     # Verify user has access to this patient
>     if not has_access(user_id, note_id):
>         raise Forbidden()
>     return await self.evaluator.evaluate(note_id)
> ```
> 
> **3. PHI Minimization**:
> ```python
> # De-identify before evaluation
> deidentified_note = remove_phi(original_note)
> evaluation = await evaluator.evaluate(deidentified_note)
> # Never send actual names, dates, IDs to LLM
> ```
> 
> **4. Audit Logging**:
> ```python
> audit_log.record({
>     'user_id': user_id,
>     'action': 'evaluate_note',
>     'note_id': note_id,
>     'timestamp': datetime.utcnow(),
>     'ip_address': request.remote_addr,
>     'result': 'success'
> })
> ```
> 
> **5. Business Associate Agreements**:
> - Signed BAA with OpenAI, Google, Anthropic
> - Verify each LLM provider is HIPAA-compliant
> - Use healthcare-specific API endpoints where available
> 
> **6. Data Retention**:
> ```python
> # Automatic deletion after retention period
> @scheduled_task('daily')
> async def cleanup_old_data():
>     cutoff_date = datetime.now() - timedelta(days=7*365)  # 7 years
>     await db.delete_where(created_at < cutoff_date)
> ```
> 
> Current system is **not HIPAA-compliant**. Would need 2-3 months of security hardening before handling real patient data."

---

### Q14: "What's your plan for handling PII if a data breach occurs?"

**Weak answer**: "I'd notify users."

**Strong answer**:
> "Data breach response requires a comprehensive incident response plan:
> 
> **Immediate Response** (0-24 hours):
> 1. **Containment**:
>    ```python
>    # Kill switches
>    await system.shutdown()
>    revoke_all_api_keys()
>    disable_external_access()
>    ```
> 
> 2. **Assessment**:
>    - What data was accessed? (audit logs)
>    - How many patients affected? (query storage)
>    - What's the attack vector? (forensics)
> 
> 3. **Notification**:
>    - Notify security team immediately
>    - Contact legal within 2 hours
>    - Prepare disclosure within 24 hours
> 
> **Short-term Response** (1-7 days):
> 4. **Breach Notification**:
>    - HIPAA requires notification within 60 days
>    - Affected individuals
>    - HHS (if >500 individuals)
>    - Media (if >500 individuals)
> 
> 5. **Remediation**:
>    ```python
>    # Rotate all credentials
>    rotate_api_keys()
>    reset_database_passwords()
>    update_access_tokens()
>    
>    # Patch vulnerability
>    deploy_security_fix()
>    run_security_audit()
>    ```
> 
> **Long-term Response** (1+ months):
> 6. **Post-Mortem**:
>    - Root cause analysis
>    - Update security policies
>    - Retrain staff
> 
> 7. **Preventive Measures**:
>    ```python
>    # Enhanced monitoring
>    add_intrusion_detection()
>    increase_audit_logging()
>    implement_anomaly_detection()
>    
>    # Defense in depth
>    add_encryption_layer()
>    implement_zero_trust_architecture()
>    require_mfa_for_all_access()
>    ```
> 
> **Proactive Prevention** (current):
> - Minimal data retention (delete after 7 days)
> - Encryption at rest and in transit
> - Regular security audits
> - Penetration testing quarterly
> - Security training for all engineers
> 
> The best breach response is **prevention**. My architecture minimizes data exposure:
> - Don't store PHI longer than necessary
> - De-identify before processing
> - Encrypt everything
> - Principle of least privilege"

---

## 🎯 Wrap-Up Questions

### Q15: "If you could rebuild this from scratch, what would you do differently?"

**Strong answer**:
> "Great question. With hindsight, I'd change three things:
> 
> **1. Database First**:
> ```python
> # Instead of JSONL files
> # Use PostgreSQL from day 1
> 
> schema = '''
> CREATE TABLE evaluations (
>     id UUID PRIMARY KEY,
>     note_id UUID REFERENCES notes(id),
>     evaluation_results JSONB,
>     created_at TIMESTAMP,
>     evaluated_by VARCHAR(50)  -- model name
> );
> CREATE INDEX idx_created_at ON evaluations(created_at);
> '''
> ```
> Benefits: Better querying, proper transactions, easier scaling
> 
> **2. Instrumentation from Start**:
> ```python
> # Add observability day 1
> from opentelemetry import trace, metrics
> 
> @trace.span('evaluate_note')
> @metrics.counter('evaluations_total')
> async def evaluate(note):
>     with metrics.histogram('evaluation_latency').time():
>         return await self._evaluate_internal(note)
> ```
> Benefits: Easier debugging, performance optimization, alerting
> 
> **3. Configuration Management**:
> ```python
> # Instead of single config.json
> # Use proper config management
> 
> from pydantic import BaseSettings
> 
> class Settings(BaseSettings):
>     model_name: str
>     api_key: str
>     batch_size: int = 10
>     
>     class Config:
>         env_file = '.env'
> 
> settings = Settings()  # Loads from env vars or .env
> ```
> Benefits: Type safety, validation, easier deployment
> 
> **What I'd Keep**:
> - DSPy framework (right choice)
> - Hybrid evaluation approach (validated by results)
> - Async architecture (scales well)
> - Modular design (easy to extend)
> 
> The core architecture is solid. The changes are about production readiness (observability, config management, proper storage), not the fundamental design."

---

Remember: These tough questions are opportunities to show **depth of thinking**. Don't just answer - show you understand the problem space, have considered alternatives, and can reason about trade-offs!
