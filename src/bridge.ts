import { MCPClient } from "./mcp-client";
import { LLMClient } from "./llm-client";
import { logger } from "./logger";
import { BridgeConfig, Tool, ServerParameters } from "./types";
import { DynamicToolRegistry } from "./tool-registry";

interface MCPMap {
  [key: string]: MCPClient;
}

export interface MCPLLMBridge {
  tools: any[];
  llmClient: LLMClient;
  initialize(): Promise<boolean>;
  processMessage(message: string): Promise<string>;
  setTools(tools: any[]): Promise<void>;
  close(): Promise<void>;
}

export class MCPLLMBridge implements MCPLLMBridge {
  private config: BridgeConfig;
  private mcpClients: MCPMap = {};
  private toolToMcp: { [toolName: string]: MCPClient } = {};
  private toolRegistry: DynamicToolRegistry;
  public llmClient: LLMClient;
  public tools: any[] = [];

  constructor(private bridgeConfig: BridgeConfig) {
    this.config = bridgeConfig;
    // Primary MCP client
    this.mcpClients["primary"] = new MCPClient(bridgeConfig.mcpServer);
    this.llmClient = new LLMClient(bridgeConfig.llmConfig);
    this.toolRegistry = new DynamicToolRegistry();

    // Initialize other MCP clients if available
    if (bridgeConfig.mcpServers) {
      Object.entries(bridgeConfig.mcpServers).forEach(([name, config]) => {
        if (name !== bridgeConfig.mcpServerName) {
          // Skip primary as it's already initialized
          this.mcpClients[name] = new MCPClient(config);
        }
      });
    }
  }

  async initialize(): Promise<boolean> {
    try {
      logger.info("Connecting to MCP servers...");

      // Initialize all MCP clients
      for (const [name, client] of Object.entries(this.mcpClients)) {
        logger.info(`Connecting to MCP: ${name}`);
        await client.connect();

        const mcpTools = await client.getAvailableTools();
        logger.info(`Received ${mcpTools.length} tools from ${name}`);

        // Register tools and map them to this MCP
        mcpTools.forEach((tool) => {
          this.toolRegistry.registerTool(tool);
          this.toolToMcp[tool.name] = client;
          logger.debug(`Registered tool ${tool.name} from ${name}`);
        });

        // Convert and add to tools list
        const convertedTools = this.convertMCPToolsToOpenAIFormat(mcpTools);
        this.tools.push(...convertedTools);
      }

      // Set tools in LLM client
      this.llmClient.tools = this.tools;
      this.llmClient.setToolRegistry(this.toolRegistry);

      logger.info(`Initialized with ${this.tools.length} total tools`);
      logger.debug(
        "Available tools:",
        this.tools.map((t) => t.function.name).join(", "),
      );

      return true;
    } catch (error: any) {
      logger.error(
        `Bridge initialization failed: ${error?.message || String(error)}`,
      );
      return false;
    }
  }

  private convertMCPToolsToOpenAIFormat(mcpTools: Tool[]): any[] {
    return mcpTools.map((tool) => {
      const converted = {
        type: "function",
        function: {
          name: tool.name,
          description: tool.description || `Use the ${tool.name} tool`,
          parameters: {
            type: "object",
            properties: tool.inputSchema?.properties || {},
            required: tool.inputSchema?.required || [],
          },
        },
      };
      logger.debug(
        `Converted tool ${tool.name}:`,
        JSON.stringify(converted, null, 2),
      );
      return converted;
    });
  }

  async processMessage(message: string): Promise<string> {
    try {
      const detectedTool = this.toolRegistry.detectToolFromPrompt(message);
      logger.info(`Detected tool: ${detectedTool}`);

      if (detectedTool) {
        const instructions =
          this.toolRegistry.getToolInstructions(detectedTool);
        if (instructions) {
          this.llmClient.systemPrompt = instructions;
          logger.debug("Using tool-specific instructions:", instructions);
        }
      }

      logger.info("Sending message to LLM...");
      let response = await this.llmClient.invokeWithPrompt(message);
      logger.info(`LLM response received, isToolCall: ${response.isToolCall}`);
      logger.debug("Raw LLM response:", JSON.stringify(response, null, 2));

      // Add tracking for tool call iterations
      let iterationCount = 0;
      const maxIterations = 3; // Maximum number of tool call iterations
      const seenToolCalls = new Set(); // Track seen tool calls to detect loops
      let allToolResults: any[] = []; // Store all tool results for final compilation

      while (
        response.isToolCall &&
        response.toolCalls?.length &&
        iterationCount < maxIterations
      ) {
        // Check for repeated tool calls
        const currentToolCall = JSON.stringify(
          response.toolCalls.map((tc) => ({
            name: tc.function.name,
            args: tc.function.arguments,
          })),
        );

        if (seenToolCalls.has(currentToolCall)) {
          logger.warn("Detected repeated tool call pattern, breaking the loop");

          // Compile all the tool results we've collected so far
          let compiledResults = "";
          if (allToolResults.length > 0) {
            compiledResults = "Here are the search results I've found:\n\n";
            allToolResults.forEach((result, index) => {
              compiledResults += `Result ${index + 1}: ${result.output}\n\n`;
            });
          }

          // Force the LLM to give a final answer with the compiled results
          const finalPrompt = `${message}\n\n${compiledResults}\n\nBased on these search results, please provide a comprehensive answer. Do not make another search request.`;
          logger.info("Sending final prompt with compiled results");
          const finalResponse =
            await this.llmClient.invokeWithPrompt(finalPrompt);
          return finalResponse.content;
        }

        seenToolCalls.add(currentToolCall);
        iterationCount++;

        logger.info(
          `Processing tool calls iteration ${iterationCount}/${maxIterations}`,
        );
        const toolResponses = await this.handleToolCalls(response.toolCalls);
        logger.info("Tool calls completed, sending results back to LLM");

        // Store all tool responses for potential final compilation
        allToolResults = allToolResults.concat(toolResponses);

        // Add specific instruction to encourage final answer after tool call
        if (iterationCount >= maxIterations - 1) {
          // On the last allowed iteration, add instruction to finalize
          const finalToolResponses = [...toolResponses];
          finalToolResponses.push({
            role: "system",
            content:
              "This is your last opportunity to use tools. After processing these results, please provide a final answer to the user's question without requesting additional tool calls.",
          });
          response = await this.llmClient.invoke(finalToolResponses);
        } else {
          response = await this.llmClient.invoke(toolResponses);
        }
      }

      // If we've hit the iteration limit but still getting tool calls
      if (response.isToolCall && response.toolCalls?.length) {
        logger.warn(
          `Reached maximum tool call iterations (${maxIterations}), forcing final answer`,
        );

        // Compile all the tool results we've collected
        let compiledResults = "";
        if (allToolResults.length > 0) {
          compiledResults = "Here are the search results I've found:\n\n";
          allToolResults.forEach((result, index) => {
            compiledResults += `Result ${index + 1}: ${result.output}\n\n`;
          });
        }

        // Force the LLM to give a final answer with the compiled results
        const finalPrompt = `${message}\n\n${compiledResults}\n\nBased on these search results, please provide a comprehensive answer. Do not make another search request.`;
        logger.info("Sending final prompt with compiled results");
        const finalResponse =
          await this.llmClient.invokeWithPrompt(finalPrompt);
        return finalResponse.content;
      }

      return response.content;
    } catch (error: any) {
      const errorMsg = error?.message || String(error);
      logger.error(`Error processing message: ${errorMsg}`);
      return `Error processing message: ${errorMsg}`;
    }
  }

  private async handleToolCalls(toolCalls: any[]): Promise<any[]> {
    const toolResponses = [];

    for (const toolCall of toolCalls) {
      try {
        const requestedName = toolCall.function.name;
        logger.debug(`[MCP] Looking up tool name: ${requestedName}`);

        // Get appropriate MCP client for this tool
        const mcpClient = this.toolToMcp[requestedName];
        if (!mcpClient) {
          throw new Error(`No MCP found for tool: ${requestedName}`);
        }

        logger.info(`[MCP] About to call MCP tool: ${requestedName}`);
        let toolArgs = JSON.parse(toolCall.function.arguments);
        logger.info(
          `[MCP] Tool arguments prepared: ${JSON.stringify(toolArgs)}`,
        );

        const mcpCallPromise = mcpClient.callTool(requestedName, toolArgs);
        const timeoutPromise = new Promise((_, reject) => {
          setTimeout(
            () => reject(new Error("MCP call timed out after 30 seconds")),
            30000,
          );
        });

        logger.info(`[MCP] Sending call to MCP...`);
        const result = await Promise.race([mcpCallPromise, timeoutPromise]);
        logger.info(`[MCP] Received response from MCP`);
        logger.debug(`[MCP] Tool result:`, result);

        toolResponses.push({
          tool_call_id: toolCall.id,
          output: typeof result === "string" ? result : JSON.stringify(result),
        });
      } catch (error: any) {
        logger.error(`[MCP] Tool execution failed with error:`, error);
        toolResponses.push({
          tool_call_id: toolCall.id,
          output: `Error: ${error?.message || String(error)}`,
        });
      }
    }

    return toolResponses;
  }

  async setTools(tools: any[]): Promise<void> {
    this.tools = tools;
    this.llmClient.tools = tools;
    this.toolRegistry = new DynamicToolRegistry();

    tools.forEach((tool) => {
      if (tool.function) {
        this.toolRegistry.registerTool({
          name: tool.function.name,
          description: tool.function.description,
          inputSchema: tool.function.parameters,
        });
      }
    });
  }

  async close(): Promise<void> {
    for (const client of Object.values(this.mcpClients)) {
      await client.close();
    }
  }
}
