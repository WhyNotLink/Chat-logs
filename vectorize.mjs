import { InferenceClient } from "@huggingface/inference";
import fs from 'fs';

const client = new InferenceClient(process.env.HF_TOKEN);

async function run() {
    const file = './chat_history.json';
    if (!fs.existsSync(file)) return;

    const data = JSON.parse(fs.readFileSync(file, 'utf8'));

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

    fs.writeFileSync('./vectors.json', JSON.stringify(data, null, 2));
}

run().catch(console.error);
