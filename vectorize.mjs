import { InferenceClient } from "@huggingface/inference";
import fs from 'fs';

const client = new InferenceClient(process.env.HF_TOKEN);

async function run() {
    const memories = JSON.parse(fs.readFileSync('./chat_history.json', 'utf8'));

    for (let session of memories) {
        for (let item of session) {
            if (!item.embedding && item.content) {
                item.embedding = await client.featureExtraction({
                    model: "shibing624/text2vec-base-chinese",
                    inputs: item.content,
                });
                await new Promise(r => setTimeout(r, 200));
            }
        }
    }

    fs.writeFileSync('./vectors.json', JSON.stringify(memories, null, 2));
}

run().catch(console.error);
