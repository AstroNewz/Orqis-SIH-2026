-- PostgreSQL / SQLite Relational Schema for CareScan / QuOra (SIH 2026)

-- 1. Pseudonymous Patients Table
CREATE TABLE IF NOT EXISTS patients (
    id VARCHAR(36) PRIMARY KEY,
    clinic_id VARCHAR(64) NOT NULL DEFAULT 'default_clinic',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN NOT NULL DEFAULT TRUE
);

CREATE INDEX IF NOT EXISTS idx_patients_clinic ON patients(clinic_id);

-- 2. Screenings Table
CREATE TABLE IF NOT EXISTS screenings (
    id VARCHAR(36) PRIMARY KEY,
    patient_id VARCHAR(36) NOT NULL,
    image_path VARCHAR(512) NOT NULL,
    scan_type VARCHAR(64) NOT NULL DEFAULT 'Intra-oral Scan',
    status VARCHAR(32) NOT NULL DEFAULT 'PENDING',
    smoking_history BOOLEAN DEFAULT FALSE,
    alcohol_consumption BOOLEAN DEFAULT FALSE,
    betel_quid BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (patient_id) REFERENCES patients(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_screenings_patient ON screenings(patient_id);
CREATE INDEX IF NOT EXISTS idx_screenings_status ON screenings(status);
CREATE INDEX IF NOT EXISTS idx_screenings_created_at ON screenings(created_at);

-- 3. Screening Results Table
CREATE TABLE IF NOT EXISTS screening_results (
    id VARCHAR(36) PRIMARY KEY,
    screening_id VARCHAR(36) NOT NULL UNIQUE,
    risk_level VARCHAR(64) NOT NULL,
    details TEXT NOT NULL,
    classical_probability REAL,
    quantum_probability REAL,
    final_probability REAL NOT NULL,
    threshold REAL NOT NULL DEFAULT 0.50,
    classification VARCHAR(64) NOT NULL,
    model_version VARCHAR(32) NOT NULL DEFAULT 'v1.0.0-qml',
    quantum_qubits INTEGER DEFAULT 8,
    quantum_shots INTEGER DEFAULT 1024,
    execution_time_ms REAL,
    is_mock BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (screening_id) REFERENCES screenings(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_results_screening ON screening_results(screening_id);
CREATE INDEX IF NOT EXISTS idx_results_final_prob ON screening_results(final_probability);

-- 4. Audit Trail Table
CREATE TABLE IF NOT EXISTS audit_logs (
    id VARCHAR(36) PRIMARY KEY,
    action VARCHAR(64) NOT NULL,
    entity_type VARCHAR(64) NOT NULL,
    entity_id VARCHAR(36) NOT NULL,
    actor_id VARCHAR(64) NOT NULL DEFAULT 'system',
    details TEXT,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_entity ON audit_logs(entity_type, entity_id);
