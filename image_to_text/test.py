import openai;
const openai = new OpenAI({
    baseURL: 'http://localhost:11434/v1',
    apiKey: 'ollama', // resquired but unused
});
const completion = await openai.chat.completions.create({
    model: 'llama2', // You can use other opensourace language models by importing and specifying their names.
    messages: [{
        role: 'user',
        content: 'What is Emotional prompt?'
    }]
});
console.log(completion.choices[0].message.content);