import { InferenceClient } from "@huggingface/inference";
import fs from 'fs';

const client = new InferenceClient(process.env.HF_TOKEN);

async function run() {
    const memoryFile = './chat_history.json';
    const memories = JSON.parse(fs.readFileSync(memoryFile, 'utf8'));

    for (let item of memories) {
        if (!item.embedding) {
            console.log(`正在转换: ${item.content}`);
            const output = await client.featureExtraction({
                model: "shibing624/text2vec-base-chinese",
                inputs: item.content,
            });
            item.embedding = output;
        }
    }

    fs.writeFileSync('./vectors.json', JSON.stringify(memories, null, 2));
    console.log("向量文件 vectors.json 已更新");
}

run().catch(console.error);
