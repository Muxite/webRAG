import { Hono } from 'npm:hono';
import { cors } from 'npm:hono/cors';
import { logger } from 'npm:hono/logger';
import { createClient } from 'npm:@supabase/supabase-js@2';

const app = new Hono();

app.use('*', cors());
app.use('*', logger(console.log));

const supabase = createClient(
  Deno.env.get('SUPABASE_URL')!,
  Deno.env.get('SUPABASE_SERVICE_ROLE_KEY')!
);

app.post('/make-server-41ae29ad/signup', async (c) => {
  try {
    const { email, password } = await c.req.json();
    
    const { data, error } = await supabase.auth.admin.createUser({
      email,
      password,
      email_confirm: true
    });

    if (error) {
      console.log(`Signup error: ${error.message}`);
      return c.json({ error: error.message }, 400);
    }

    return c.json({ user: data.user });
  } catch (error) {
    console.log(`Signup exception: ${error}`);
    return c.json({ error: 'Signup failed' }, 500);
  }
});

app.post('/make-server-41ae29ad/scrape', async (c) => {
  try {
    const accessToken = c.req.header('Authorization')?.split(' ')[1];
    const { data: { user }, error } = await supabase.auth.getUser(accessToken);
    
    if (!user?.id) {
      return c.json({ error: 'Unauthorized' }, 401);
    }

    const { prompt } = await c.req.json();
    
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    return c.json({ 
      result: `Scraped results for: "${prompt}"\n\nThis is where your AI scraper results would appear.` 
    });
  } catch (error) {
    console.log(`Scrape error: ${error}`);
    return c.json({ error: 'Scrape failed' }, 500);
  }
});

Deno.serve(app.fetch);
