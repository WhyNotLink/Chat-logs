import { InferenceClient } from "@huggingface/inference";
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const client = new InferenceClient(process.env.HF_TOKEN);
const __dirname = path.dirname(fileURLToPath(import.meta.url));

async function run() {
    const filePath = path.join(__dirname, 'chat_history.json');
    const outputPath = path.join(__dirname, 'vectors.json');

    if (!fs.existsSync(filePath)) return;

    const data = JSON.parse(fs.readFileSync(filePath, 'utf8'));

    for (let session of data) {
        for (let msg of session) {
            if (!msg.embedding && msg.content) {
                console.log(`Vectorizing: ${msg.content.substring(0, 15)}`);
                try {
                    msg.embedding = await client.featureExtraction({
                        model: "shibing624/text2vec-base-chinese",
                        inputs: msg.content,
                    });
                    await new Promise(r => setTimeout(r, 200));
                } catch (e) {
                    console.error(e.message);
                }
            }
        }
    }

    fs.writeFileSync(outputPath, JSON.stringify(data, null, 2));
}

run().catch(console.error);
