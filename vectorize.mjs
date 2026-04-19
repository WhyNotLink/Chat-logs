import { InferenceClient } from "@huggingface/inference";
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const client = new InferenceClient(process.env.HF_TOKEN);
const __dirname = path.dirname(fileURLToPath(import.meta.url));

async function run() {
    const filePath = path.join(__dirname, 'chat_history.json');
    const outputPath = path.join(__dirname, 'vectors.json');

    if (!fs.existsSync(filePath)) {
        console.log("File not found: chat_history.json");
        return;
    }

    const data = JSON.parse(fs.readFileSync(filePath, 'utf8'));
    let updatedCount = 0;

    for (let session of data) {
        for (let msg of session) {
            // 只有当没有向量且有内容时才处理
            if (!msg.embedding && msg.content) {
                console.log(`Vectorizing: ${msg.content.substring(0, 15)}...`);
                try {
                    msg.embedding = await client.featureExtraction({
                        model: "shibing624/text2vec-base-chinese",
                        inputs: msg.content,
                    });
                    updatedCount++;
                    // 稍微停顿，避免触发 HF 的频率限制
                    await new Promise(r => setTimeout(r, 200));
                } catch (e) {
                    console.error(`Error for "${msg.content.substring(0, 10)}": ${e.message}`);
                }
            }
        }
    }

    fs.writeFileSync(outputPath, JSON.stringify(data, null, 2));
    console.log(`Successfully processed ${updatedCount} new messages.`);
}

run().catch(console.error);
