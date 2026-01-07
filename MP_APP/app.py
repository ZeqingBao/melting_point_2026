from flask import Flask, render_template, request, send_from_directory, jsonify
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Crippen, Draw
import io
import base64
import os
import csv
import uuid


# Configuration
PORT = 5001

app = Flask(__name__)

# Temporary directory for batch outputs
TMP_DIR = os.path.join(os.path.dirname(__file__), 'tmp')
os.makedirs(TMP_DIR, exist_ok=True)

# How many rows to show in the preview tables
PREVIEW_LIMIT = 50


def compute_properties(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    logp = Crippen.MolLogP(mol)
    mw = Descriptors.MolWt(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)

    try:
        img = Draw.MolToImage(mol, size=(300, 300))
        bio = io.BytesIO()
        img.save(bio, format='PNG')
        data = base64.b64encode(bio.getvalue()).decode('utf-8')
        img_uri = f"data:image/png;base64,{data}"
    except Exception:
        img_uri = None

    props = {
        'logp': round(logp, 3),
        'mw': round(mw, 3),
        'tpsa': round(tpsa, 3),
        'hbd': int(hbd),
        'hba': int(hba),
        'img': img_uri,
    }
    return props


def placeholder_melting_point(props):
    mw = props.get('mw', 0)
    logp = props.get('logp', 0)
    tpsa = props.get('tpsa', 0)
    mp = 0.15 * mw + 4.0 * logp - 0.05 * tpsa + 20
    return round(mp, 2)


@app.route('/', methods=['GET', 'POST'])
def index():
    error = None
    result = None
    if request.method == 'POST':
        smiles = request.form.get('smiles', '').strip()
        if not smiles:
            error = 'Please provide a SMILES string.'
        else:
            props = compute_properties(smiles)
            if props is None:
                error = 'Invalid SMILES string.'
            else:
                mp = placeholder_melting_point(props)
                result = {'smiles': smiles, 'props': props, 'mp': mp}

    return render_template('index.html', error=error, result=result)


@app.route('/batch', methods=['POST'])
def batch():
    uploaded = request.files.get('file')
    if not uploaded:
        return render_template('index.html', error='No file uploaded')

    uid = uuid.uuid4().hex
    in_path = os.path.join(TMP_DIR, f'{uid}_in.csv')
    out_name = f'{uid}_out.csv'
    out_path = os.path.join(TMP_DIR, out_name)
    uploaded.save(in_path)

    output_rows = []
    input_rows = []
    with open(in_path, newline='', encoding='utf-8') as fh:
        reader = csv.reader(fh)
        rows = list(reader)
        start = 0
        if rows:
            first = rows[0][0] if rows[0] else ''
            if 'smiles' in first.lower() or 'smile' in first.lower():
                start = 1

        for r in rows[start:]:
            if not r:
                continue
            s = r[0].strip()
            input_rows.append([s])
            if not s:
                output_rows.append([s, ''])
                continue
            props = compute_properties(s)
            if props is None:
                mp = ''
            else:
                mp = placeholder_melting_point(props)
            output_rows.append([s, mp])

    # Write output CSV
    with open(out_path, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh)
        writer.writerow(['smiles', 'predicted_melting_point'])
        for r in output_rows:
            writer.writerow(r)
    preview_in = input_rows[:PREVIEW_LIMIT]
    preview_out = output_rows[:PREVIEW_LIMIT]
    return render_template('index.html', batch_in=preview_in, batch_out=preview_out, batch_file=out_name)


@app.route('/batch_ajax', methods=['POST'])
def batch_ajax():
    uploaded = request.files.get('file')
    if not uploaded:
        return jsonify({'error': 'No file uploaded'}), 400

    uid = uuid.uuid4().hex
    in_path = os.path.join(TMP_DIR, f'{uid}_in.csv')
    out_name = f'{uid}_out.csv'
    out_path = os.path.join(TMP_DIR, out_name)
    uploaded.save(in_path)

    output_rows = []
    input_rows = []
    with open(in_path, newline='', encoding='utf-8') as fh:
        reader = csv.reader(fh)
        rows = list(reader)
        start = 0
        if rows:
            first = rows[0][0] if rows[0] else ''
            if 'smiles' in first.lower() or 'smile' in first.lower():
                start = 1

        for r in rows[start:]:
            if not r:
                continue
            s = r[0].strip()
            input_rows.append([s])
            if not s:
                output_rows.append([s, ''])
                continue
            props = compute_properties(s)
            if props is None:
                mp = ''
            else:
                mp = placeholder_melting_point(props)
            output_rows.append([s, mp])

    with open(out_path, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh)
        writer.writerow(['smiles', 'predicted_melting_point'])
        for r in output_rows:
            writer.writerow(r)

    preview_in = input_rows[:PREVIEW_LIMIT]
    preview_out = output_rows[:PREVIEW_LIMIT]
    return jsonify({'preview_in': preview_in, 'preview_out': preview_out, 'file': out_name})


@app.route('/download/<path:filename>')
def download(filename):
    return send_from_directory(TMP_DIR, filename, as_attachment=True)


if __name__ == '__main__':
    # Run without the debug reloader to keep single-process handling uploads reliably.
    # Toggle debug with the FLASK_DEBUG environment variable (set to '1' for debug).
    debug = os.environ.get('FLASK_DEBUG', '0') == '1'
    app.run(host='0.0.0.0', port=PORT, debug=debug)
