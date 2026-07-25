declare module "affon:errors" {
  export interface AffonErrorConstructor {
    new(code: AffonErrorCode, message: string): AffonError
    readonly prototype: AffonError
  }

  export const AffonError: AffonErrorConstructor
  export default AffonError
}
